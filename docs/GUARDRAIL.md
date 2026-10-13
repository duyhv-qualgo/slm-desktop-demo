# Text Guardrail

On-device safety classification for chat messages. Written for whoever picks this
up next — it assumes you can read Swift and have skimmed [HANDOFF.md](HANDOFF.md),
and it assumes nothing about content moderation.

**This is a personal filter, not a scanning system.** The distinction is the whole
of [§2](#2-why-on-device), and it decides most of the design.

---

## 1. Overview

Every message in the app is screened by two things running entirely on the phone:

1. a **regex rule set** for the user's region, and
2. a **BERT-Tiny TFLite classifier** for the message's language,

producing a verdict of **class** (10 values, `Clean` plus 9 harms) and **severity**
(4 values, `Safe`…`Critical`). Critical blurs the bubble behind a reveal overlay;
Caution and Violation label it and leave it readable.

Steady state is **7–25 ms per message** on a background queue. Nothing is sent
anywhere. No LLM is involved.

What it is not:

- **Not enforcement.** Nothing is blocked, refused, or reported. A verdict changes
  how a message is displayed on this device, and that is all.
- **Not a server filter.** It cannot be, and [§2](#2-why-on-device) explains why
  that constraint turned out to be a feature.
- **Not a substitute for reporting.** There is no report or block flow yet
  ([§11](#11-known-limits)), which is a real gap for a messaging product.
- **Not the SLM.** An earlier version prompted the task model to classify
  messages; it was removed. `GuardEngine` never calls a generative model.

---

## 2. Why on-device

End-to-end encryption removes the server-side option: the server cannot read
messages, so client-side is what is left.

That path has a cautionary precedent. In **August 2021** Apple announced on-device
CSAM detection for iCloud Photos — NeuralHash matching against known hashes, with
private set intersection and threshold secret sharing so Apple learned nothing
below a match threshold. Careful work. It was delayed within a month and
**abandoned in December 2022**: *"Children can be protected without companies
combing through personal data."*

The objection was never that computation happened on the device. It was that Apple
paired on-device **detection** with off-device **reporting**, and once that pipeline
exists the match list is a policy knob — change what it matches and the same
shipped mechanism detects something else.

Our mechanism is nearly the same shape. The difference is entirely **where the
verdict goes: ours stays on the device.** That is what makes this a personal
filter rather than a scanning system.

Three rules follow, and the rest of this document assumes them:

- **Verdicts are local.** Nothing content-derived leaves the device by default.
- **Sender-side classification is advisory to the sender only.** Transmitting a
  predicted label would put us in the Apple 2021 shape, and it is also trivially
  defeated — the sender owns the device doing the accusing.
- **Reputation, if it is ever built, comes from recipient reports** — voluntary
  disclosure by the person who received the message — never from sender-side
  telemetry.

One consequence to state early: **shipping the detector ships it to the
adversary.** Researchers extracted NeuralHash from iOS and produced collisions
within days. Our rule file is *readable JSON in the app bundle*. See
[§11](#11-known-limits) for what that means the feature may claim.

---

## 3. Architecture

### 3.1 Layer stack

| Layer | File | Job |
|---|---|---|
| TFLite C API | `SLMChat/TFLiteC.h` | Vendored declarations. No separate framework — symbols resolve against the embedded `CLiteRTLM.xcframework` |
| Bridge | `SLMChat/GuardLM.h/.mm` | Obj-C++ over the C API: `load(modelPath:)` once, `classify(inputIds:attentionMask:)` per window |
| Tokenizer | `Engine/GuardTokenizer.swift` | Hand-ported BPE: NFKC + lowercase → `\w+\|[^\w\s]+` → merges over Unicode scalars |
| Rules | `Engine/GuardRules.swift` | One (region, language) regex set, compiled once, most-severe match wins |
| Packs | `Engine/GuardPackCatalog.swift` | language → (model, tokenizer) |
| Regions | `Engine/GuardRegion.swift` | jurisdiction → rule files + actionable classes |
| Engine | `Engine/GuardEngine.swift` | Serial `.utility` queue, LRU pack cache, windowing, decision logic |
| Scheduling | `Engine/MessageScreener.swift` | Which messages get screened, in what order |
| Service | `Engine/ChatService.swift` | `classify(_:language:)`, `guardrailError` |
| Call sites | `Model/Conversation.swift`, `Model/GroupChat.swift` | live and viewport paths |
| Verdict model | `Model/SafetyLabel.swift` | `SafetyLabel`, `Severity`, `SafetyResult` |

### 3.2 The two axes

The single most important structural idea here: **language and region are
orthogonal**, and both are needed to pick the right screening.

|  | Selects | Keyed by |
|---|---|---|
| **Language** | which classifier runs | the message's own language |
| **Region** | which policy applies | the *reader's* jurisdiction |

A rule encodes a **policy**, and policy is set by jurisdiction, not by the
alphabet the text happens to use. Vietnam restricts anti-government speech and
says so in 114 patterns; the same speech is protected in the UK. A Vietnamese
message can be read in Berlin — so region tracks the **reader**, never the sender.

*Detect universally, decide locally.*

Making that true took two mechanisms, and each closed a measured hole.

**Rule sets are keyed by (region × language).** A set can only match text in the
language it was written in, so region alone silently means "region, in whichever
language someone happened to author it". Measured, for a reader in London:

| | Verdict |
|---|---|
| "Move your money to a safe account straight away." | Scam / **Critical** |
| "Chuyển tiền sang tài khoản an toàn ngay lập tức." | **Clean, 0.915** |

Same scam, same reader, opposite outcomes. The UK now ships `uk-en` **and**
`uk-vi` (it serves a large Vietnamese-speaking population), and a pair with no
authored file falls back to `guardrail-rules-universal.json`.

At match time **every** set of the region is scanned, not just the one matching
the detected language. Selecting by language is the obvious reading and it is
wrong: detection abstains on short messages — the ones rules are best at — and a
wrong guess would silently route text to a set with no patterns for it. Scanning
all of them costs nothing to correctness, because a pattern only matches its own
language. **The language key organises authoring, not dispatch.**

**The model is gated by `GuardRegion.actionableClasses`.** The rules alone could
not express this: they are language-locked, so a UK reader's English patterns
never fire on a Vietnamese message whatever categories they contain. The model
has no such gap — it understands the language and applies its own 10-class
taxonomy identically everywhere, and `political` is one of those classes. Without
the mask, `"đảng bán nước"` came back **Political / Violation, 0.909** for a
reader in London, where that speech is protected. It now returns Clean there, and
still Political in Vietnam — while a genuine threat in the same conversation is
still caught in both.

Practical consequences: switching region reloads **only** the rules and the mask
(the classifier is region-invariant); a language with no model pack is not
model-screened at all, but rules still run because they scan raw text.

### 3.3 Life of one message

```
text
 ├─ language: message.detectedLanguage ?? thread language        ~0–1.5 ms
 ├─ RULES: most-severe match over the full text (20k cap)        ~6.5 ms
 │    └─ hit? → verdict(class, pattern severity, confidence 1.0)  ← short-circuits
 ├─ pack: resident, or lazily loaded                             ~0 / 50 / 320–500 ms
 ├─ windows: one sentence each + whole message if it fits        (cap 16)
 │    └─ per window: encode → invoke → softmax → argmax           ~7–11 ms
 │                    → severity clamp → 0.5 confidence gate
 └─ worst window wins → SafetyResult
```

Then, on the main actor: `message.apply(result)` publishes the verdict and
`persist()` writes (~1.5 ms even on a 1000-message thread — measured, and not the
suspect it looks like).

---

## 4. Detection

### 4.1 The rule pre-filter

Regular expressions run **before** the tokenizer and short-circuit on a hit. They
are worth keeping alongside a trained classifier for two reasons:

- **They are certain.** The model gates on a 0.5 confidence threshold, so a known
  slur can slip through at 0.49. A pattern either matches or it does not.
- **They see everything.** The tokenizer truncates at 128 tokens per window and
  windowing is capped at 16, so the model can be padded out of reach. A regex scans
  the full text up to `scanLimit` (20 000 UTF-16 units).

Rules can only ever **raise** a verdict; nothing a pattern does can clear a message
the model would have flagged. That is what makes running them first safe.

`firstMatch` returns the **most severe** hit, not the first, and stops early on
Critical. Ordering by position would let a `caution` rule in an early category
(the bare name of a banned knife) mask a `critical` one later (an offer to sell it).

### 4.2 The model

BERT-Tiny with two heads — 10-class and 4-severity — from
`kchat/2026_07_06_10_class_4_severity`. `GuardEngine.decide` routes heads **by
length** (10 = class, 4 = severity), so a checkpoint with the same shape is a
drop-in.

Classes: `clean, violence, sexual, political, childSafety, selfHarm, hate,
harassment, illegalGoods, scamFraud`. Severities: `safe, caution, violation,
critical`.

### 4.3 Windowing

Long messages are split into **one sentence per window**, plus the whole message as
a context window when it fits. Worst window wins.

The obvious design — pack sentences up to the 128-token limit — was tried and
**measured not to work**. The model judges a window by its *dominant* content:

> `"Tao sẽ giết mày."` alone → **Violence / Violation**, 0.911
> the same sentence behind one benign sentence → **Clean**, 0.905

So the unit has to be small enough that a violation dominates its own window.

Every window costs a full forward pass — `encode` pads to a fixed 128, so a
13-token sentence costs the same as a full one. Hence `maxWindows = 16` (~160 ms
worst case). A message with more sentences than that is screened only up to the
cap; **the rules are the backstop.**

### 4.4 Verdict

```
model:  softmax → argmax → per-class severity clamp → confidence ≥ 0.5
rule:   class + the PATTERN's own severity + confidence 1.0
```

The severity clamp mirrors the upstream `infer.py`: Clean is always Safe, Political
caps at Violation, Child Safety is fixed Critical, the rest span Caution–Critical.

Confidence is **1.0 for a rule hit** and is not compared against the threshold —
that gate exists to suppress the model's low-confidence guesses, and a pattern
match has no equivalent doubt. This is also how you tell the two apart in the log:
**exactly 1.000 means a rule fired.**

---

## 5. Regional rule sets

### 5.1 Why region, not language

See [§3.2](#32-the-two-axes). The test that enforces it is
`testPoliticalRulesDoNotLeakOutOfVietnam` — it fails if anyone copies Vietnam's
political patterns into a European region.

### 5.2 Build pipeline

```
guardrails/regions/<region>-<lang>.yaml   (+ _universal.yaml, merged into all)
   └─ python3 scripts/build_regex_patterns.py [--region <region>-<lang>]
        ├─ Resources/guardrail-rules-<region>-<lang>.json
        └─ Resources/guardrail-rules-universal.json   (the fallback)
             └─ xcodegen generate      ← required for a NEW file to reach the bundle
```

Every pair built here needs an entry in `GuardRegion.ruleLanguages`, or it is
never loaded. `testEveryDeclaredPairHasAFile` catches the reverse.

The script validates that every pattern compiles under Python's `re` and exits
non-zero on a bad one or an unknown category. JSON because iOS has no YAML parser.
A script because the patterns are an upstream artifact that will be re-released.

> ⚠️ `guardrails/` is **gitignored**. The YAML sources and the build script are not
> in version control; only the generated JSON ships. This should be fixed — a
> policy file is exactly the thing that needs review history.

### 5.3 What ships

| Set | Patterns | Status |
|---|---|---|
| `vietnam-vi` | 200 | authored — child 32, violence 22, sexual 32, political 114 |
| `uk-en` | 127 | authored — child 12, violence 21, self-harm 10, sexual 7, hate 22, harassment 8, illegal goods 16, scam/fraud 31 |
| `uk-vi` | 21 | authored — scam/fraud 14, violence 2, harassment 2, child 3. **Needs a native-speaker review pass** |
| `france-fr`, `germany-de`, `spain-es` | 3 each | universal baseline only, `authored: false` → Settings caption |
| `universal` | 3 | fallback for any pair with no authored file |

⚠️ `germany-de` builds but can never load: `TranslateLanguage` has no German case,
so no message routes to it. Germany runs on the universal fallback until German is
added as a language.

The two authored sets differ in **which categories they populate**, which is the
point of splitting by region:

- **Vietnam** carries 114 political patterns. The UK carries **zero**, deliberately
  — `testUKHasNoPoliticalRules` pins it. Foreign interference is a priority offence
  under the Online Safety Act, but no regular expression can separate it from
  protected criticism of a government.
- **UK** is where scam/fraud (31), self-harm, hate, harassment and illegal goods
  live. Vietnam's upstream set predates those categories.

Two invariants in `uk-en.yaml` worth not breaking:

- **`self_harm` matches only encouragement aimed at another person.** First-person
  distress must never be flagged — silencing someone reaching for help is the worst
  thing this file could do. `testFirstPersonDistressIsNotFlagged` pins seven
  phrasings.
- **No bare drug nouns.** Every drug pattern requires supply context — a price, a
  quantity, a delivery. "grabbed a coke", "the grass needs cutting" and "my gear is
  in the car" are all in the benign corpus.

**Never machine-translate patterns between regions.** Slang, idiom and euphemism do
not survive it, and you manufacture both false positives (idioms that read violent
literally) and false negatives (local euphemisms nothing matches).

Legal background for the UK set is in [UK_COMPLIANCE.md](UK_COMPLIANCE.md).

### 5.4 Per-pattern severity

A YAML entry is either a bare string (severity = the category ceiling) or a mapping:

```yaml
hate:
- \bgas\s+all\s+the\s+jews\b            # → ceiling (critical)
- re: \bn[i1]gg[a4]s?\b
  severity: caution                      # labels, does not blur
```

Lower severities exist for matches that are **evidence, not proof** — a slur with
common reclaimed in-group use, the bare name of a banned weapon, "watch your back"
(a warning as often as a threat). Blur is gated on Violation/Critical, so those
surface a label without hiding the message.

Caution is not a free tier, though. `\bi know where (you|u) live\b` shipped at
caution and was **removed** rather than downgraded further: it carries no threat
of its own, it is a punchline between friends far more often than it is a
stalker, and a safety label under an ordinary joke is still a false positive.
The nearby harassment pattern for *work / school / the pub* stays — that phrasing
has no joke shape. When a pattern's honest severity would be "probably nothing",
the answer is to delete it, not to lower it.

Every pattern used to land on its category ceiling, which held while the only rule
set was the precision-tuned Vietnamese one. The UK set broke that assumption, and
a guardrail that hides ordinary conversation stops being trusted.

> Adding a **new category** needs three edits: `ORDER` *and* `CEILING` in
> `build_regex_patterns.py`, *and* `GuardRules.labels` in Swift. Miss the Swift one
> and the JSON ships with your patterns in it while the app logs
> `unknown category` and screens for none of them.
> `testEveryBuiltCategoryIsMapped` compares declared vs. loaded counts to catch it.

---

## 6. Language packs

`GuardPackCatalog` maps language → (model resource, tokenizer resource).
`GuardEngine` keeps an LRU of `maxResidentPacks = 8` — sized to hold the whole
planned 7-language catalog.

- **~3 MB resident per pack**, not the 9.6 MB of the `.tflite`: TFLite memory-maps
  the model, so only the tensor arena and vocabulary land in the footprint.
- **Only the native language is preloaded** at warm-up. Others load lazily on first
  message in that language. Eagerly loading all seven would put ~600–750 ms into
  startup for languages the user may never receive a message in.
- A language with **no pack is not model-screened at all**. Rules still run.
  `testUnsupportedLanguageIsNotScreened` also asserts we never run one language's
  classifier over another's text — that would return a confident verdict about text
  it never understood.

`testCacheCanHoldTheWholeCatalog` fails if the catalog outgrows the cache, because
the symptom is silent per-message thrashing rather than an error.

> ⚠️ `guardrail-bert-tiny-vi.tflite` and `-en.tflite` are currently
> **byte-identical copies** of one bilingual checkpoint, shipped under per-language
> names so the pack machinery runs its real path. A verdict can never tell you
> which pack ran, so `GuardPackTests` asserts on `residentLanguages()`, not output.

---

## 7. Recipient-side: when screening runs

### 7.1 The governing principle

**Cost scales with what is read, not with what arrives.**

That matters more here than in a typical app: this is an E2EE messenger, so the
device is already spending CPU decrypting the same messages. The guardrail is the
flexible consumer and must yield.

### 7.2 Two paths

| | Trigger | Persist |
|---|---|---|
| **Live** — `classifySafety(in:)` | message arrives while the user is here | immediate |
| **Bulk** — `screenFromViewport(_:)` | row `.onAppear` | coalesced, 500 ms |

Live messages arrive at human typing speed, and the verdict lands a frame or two
after the bubble renders — imperceptible. A **backlog** is different: 500 messages
arriving at once, for the handful anyone actually scrolls to. Those are appended
**unscreened** and wait for the viewport.

Measured: a 500-message flood costs **0** classifications until opened, and **11**
when it is.

The one exception: **the newest message of a batch is always screened eagerly**,
because the inbox row renders it ([§7.4](#74-the-surfaces-that-cannot-wait)).

### 7.3 `MessageScreener`

Shared by `Conversation` and `GroupChat` — the scheduling is identical and too easy
to get subtly wrong twice. It replaces a task-per-message drained FIFO, which had
two defects that only appeared under load:

- **Bidirectional look-ahead** (±10). Older-only was correct while chats opened at
  the bottom. They now open at the **first unread message**, so the unread ones are
  *below* and the reader travels **down** — one-directional look-ahead would race
  away from the direction of travel.
- **Nearest-to-viewport first.** One drain loop over a pending set; the next
  message is chosen *when the previous finishes*, so scrolling mid-drain changes
  what runs next. Under FIFO, the message on screen waited behind ~200 rows the
  reader had already passed.
- **Shedding past 48 pending.** A shed message is **forgotten** — removed from the
  claimed set — so scrolling back re-requests it through `onAppear`. It is not
  marked done; that would silently lose coverage.
  `testShedMessagesAreScreenedWhenScrolledBackTo` pins the difference.

Before and after, on a 200-row flick through a 300-message group:

| | Before | After |
|---|---|---|
| Queued | 210 | **48** (capped) |
| Done when the scroll ended | 17 | **48** |

Look-ahead buys **timeliness, not savings** — total work still equals the messages
you actually scroll past. It just moves that work ~10 rows earlier so the blur is
already there when you arrive.

### 7.4 The surfaces that cannot wait

Deferral is the whole performance win, but some surfaces render message text
*before* the chat view does, and they must respect the verdict or the guardrail
leaks around itself:

1. **The viewport** — handled by look-ahead.
2. **The inbox preview** — `lastMessagePreview` substitutes `⚠️ Class · Severity`
   for anything the thread would blur. It deliberately ignores `revealUnsafe`:
   revealing inside a thread is a choice for that screen and that moment, not
   permission to republish.
3. **Push notifications** — none exist yet. When they do, they need the same check,
   and they are worse: content on a lock screen, outside the app entirely.

Anything future that renders message text — search results, widgets, share sheets —
needs a case in `ContentLeakTests`.

### 7.5 Techniques

| Technique | Effect | Status |
|---|---|---|
| Defer bulk to the viewport | 500-message backlog → 0 until opened | built |
| Bidirectional look-ahead (±10) | verdict exists before the row is read | built |
| Nearest-viewport-first ordering | 210 queued/17 done → 48/48 | built |
| Shed past 48 pending | bounded queue; shed rows re-screen on return | built |
| Rules-first short-circuit | a hit skips tokenizer *and* model | built |
| Classify once, persist | history never re-screened | built |
| `.utility` serial queue | never contends the UI or the decrypt path | built |
| Lazy packs, LRU 8 | ~3 MB each, ~50 ms warm | built |
| Coalesced persist | 11 verdicts → 1 write | built |
| Content-hash dedup | forwards/broadcasts screened once | planned |
| Literal prescan before regex | recovers ~6.5 ms on the common path | planned |
| Thermal / Low Power / background → rules-only | degrade, don't skip | planned |
| Explicit backpressure from decrypt | QoS alone doesn't guarantee yield | planned |

---

## 8. What the user sees

- **Blur only Critical.** Caution and Violation are labelled and stay readable. The
  rule lives once, on `ChatMessage.isSensitive`, because the bubble and the inbox
  preview both ask it and two copies would drift silently.

  This was Violation-and-above until a user wrote first-person distress —
  *"mình thấy mệt và trống rỗng quá… nếu mình biến mất thì mọi thứ sẽ nhẹ nhàng
  hơn"* — and the model returned Self-harm/Violation, so the app **blurred a cry
  for help**. The §7.2 invariant ("first-person distress must never be flagged")
  only ever bound the hand-authored rules; the model has no such constraint.

  Severity turned out to separate the two cases without a class special-case:
  across 13 samples, first-person distress came back Clean (9/10) or Violation
  (1/10) and never Critical, while encouragement aimed at another person
  (*"You should just kill yourself"*) came back Critical every time.

  What it costs: `severityRange` caps `political` at Violation, so political
  content can no longer be blurred at all — retiring 114 Vietnamese political
  patterns from hiding anything, consistent with the position on political speech
  in [UK_COMPLIANCE.md §3](UK_COMPLIANCE.md). Elsewhere it is 3 of 127 UK
  patterns; child safety is pinned to Critical by `severityRange` and always
  blurs.
- **Reveal is one tap**, per message, and the overlay says which class and severity.
- **Report — long-press any message from someone else.** It lives in the context
  menu beside Copy and Summarize, marked destructive so it sorts last; a reported
  message then carries a "Reported" marker in the bubble.

  It used to be an inline button restricted to Critical, and the restriction was
  wrong. The radicalisation scenario shows why: the classifier catches 2 of its 8
  lines and returns Clean for the recruitment (*"Cần tổ chức ngay. Ai dám tham gia
  thì khai tên ra."* 0.916) and the purge-organising — so a Critical-only gate
  refused a report on exactly the six lines a person would want escalated. **A
  report is not the classifier's judgement**, it is a human saying someone should
  look at this, so gating it on the classifier agreeing gets the dependency
  backwards. The only rule left is that you cannot report your own message.

  Reports on unflagged messages are therefore normal, and the payload marks them:
  `reasonMatchesVerdict` is `nil` rather than a false disagreement, and
  `isMissedByGuardrail` flags the set worth reviewing first — both for moderation
  and for the next pack. The log prints `GUARDRAIL MISS` for them.

  Tapping it opens a **report sheet** rather than submitting: a reason dropdown
  pre-filled from the verdict, an optional 300-character note, and a plain
  statement of what gets sent. Content the thread was blurring stays blurred in the
  sheet, with its own "Show message" — opening the form must not do what the blur
  was for. Content that was already readable is shown, since re-hiding it is
  theatre and costs the user the context they need to pick a reason.

  Two things the sheet carries that a one-tap report can't. The reason is
  **pre-filled but owned by the reporter**, and when they change it the sheet says
  so ("you've picked something different from what we detected"); that
  disagreement is a labelled example of the model being wrong, volunteered by
  someone who read the message, arriving from real traffic rather than a test set.
  And `political` is **absent from the reason list**, permanently — Vietnam ships
  114 political patterns and the UK ships zero because political criticism is
  protected expression, and a "political content" reason would hand every user a
  button for silencing dissent.

  The report is the **only** thing in this design that may leave the device, and
  it leaves because a human chose to send it — never because the classifier fired.
  That is the line between a personal filter and client-side scanning (§2). It
  carries the class, severity, author and detection source, and deliberately not
  the message text. ⚠️ It logs; there is no backend, and no franking, so a real
  submission would be an unverifiable claim — see `GuardReport` for what is missing.

- **Blurred content is hidden from VoiceOver** (`.accessibilityHidden`). A blur is a
  visual effect; without this, assistive technology reads the hidden text aloud
  verbatim. The reveal button stays reachable, so the content is still available —
  deliberately, the same as for a sighted user. Text selection is blocked too.
- **Never flag the victim.** The distressed replies of someone being harassed must
  come back clean. This is a content-authoring constraint enforced by test, not a
  UI behaviour.
- **Fail visibly.** The guardrail fails open to `.clean` everywhere, so a broken
  load looks exactly like "all clean". `ChatService.guardrailError` →
  `ConversationStore.guardrailError` surfaces as a ChatView banner and a Settings
  "Safety" section showing region, rule count, and whether the region is authored.
- **Groups: collapse, don't wall.** A flooded group with 12 flagged messages
  currently shows twelve blurred bricks. "12 messages hidden — show" would read
  better and render cheaper. *Not built.*

---

## 9. Performance

All figures are the **iPhone 17 simulator**, which runs on the host Mac's CPU, and
they are noisy. The stated low-end target is an **iPhone XS** (A12, CPU-only —
Metal needs A13+). Expect meaningfully slower there; it has not been measured, and
that measurement is the first thing to do before trusting any of this on device.

| | |
|---|---|
| Steady-state short message | 7–25 ms |
| Language detection | 1.5 ms uncached, ~0 cached |
| Rules scan (200 patterns, full text) | ~6.5 ms |
| One inference window | 7–11 ms |
| 300-token message, 3 / 16 windows | 20 ms / 236 ms |
| First inference on a fresh interpreter | ~100 ms, then ~24 ms |
| Pack load — warm / cold | ~50 ms / 320–500 ms |
| Sustained throughput | 7–10 ms/msg, ~135 msg/s |
| `apply` + `persist` on a 1000-message thread | ~1.5 ms |
| Resident RAM per pack | ~3 MB |

**Cost scales with sentence count, not length** — every window is a full forward
pass padded to 128 tokens.

### Three clocks, easily confused

```
[Guard] [English] "…" → Sexual/Critical conf=0.804 (FLAGGED) 39.1ms
[GuardEngine] 4.5ms work, 32.7ms queued, 12 behind
[GuardDemo] 12/13  935.3ms  …
```

| Tag | Measures |
|---|---|
| `[Guard]` | the classifier, from the caller's side |
| `[GuardEngine]` | on-queue split: work vs. time spent waiting behind other messages |
| `[GuardDemo]` | **wall clock**, including the wait to resume on the main actor |
| `[GuardPerf]` | the main-actor tail (`apply` + `persist`), only when it exceeds a frame |

A 39 ms classification once logged as 935 ms. The gap was **main-actor
contention** — SwiftUI relaying out the list and rasterizing a blur, in a Debug
build with a debugger attached — not guardrail work. The log said so itself:
`Hang detected: 0.91s`.

> **Backlog depth must be read from `pendingScreenings`, not
> `GuardEngine.backlogDepth`.** Enqueues are Swift `Task`s that have not reached the
> engine's dispatch queue until the main actor yields, so the engine reports **0**
> while hundreds of messages are genuinely waiting one layer up. Both are logged so
> the discrepancy is visible rather than confusing.

---

## 10. Testing and demos

**129 tests.** The guardrail-specific suites:

| Suite | Covers |
|---|---|
| `GuardEngineTests` | parity with the original Python model (`classifyModelOnly`, rules bypassed) |
| `GuardTokenizerTests` | 70 fixtures vs. the real HF tokenizer |
| `GuardRulesTests` | compile counts, reference categories, benign idioms, region isolation |
| `GuardUKRulesTests` | UK detections, benign UK English, severity tiers, no political rules |
| `GuardWindowTests` | sentence windows, the caps, violations past the first window |
| `GuardPackTests` | catalog, lazy load, LRU, unsupported languages |
| `ViewportScreeningTests` | live vs. bulk split, look-ahead, double-screening |
| `MessageScreenerTests` | bidirectional look-ahead, ordering, shedding |
| `GroupScreeningTests` | group wiring, per-author attribution, pile-on concentration |
| `GroupFloodTests` | flood cost, throughput, verdict stability under load |
| `ScreeningTailTests` | main-actor cost after a verdict |
| `ContentLeakTests` | hidden content not printed elsewhere |
| `GuardrailDemoTests` | the scripted scenarios still demonstrate what they claim |

```bash
xcodebuild test -scheme SLMChat \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:GuardrailTests
```

### Demos

The 🛡️ toolbar button in both chat views replays scripted conversations at
~1.2 s per line:

The ✨ menu is split by thread type — scenarios whose lines name an author
appear only in a group, since playing them 1:1 collapses every author into one
anonymous "friend" and destroys the attribution they exist to show.

| Scenario | Shows | Where |
|---|---|---|
| Harassment → sextortion | the full severity ladder; clean → caution → critical | 1:1 |
| UK scam / fraud | the highest-volume UK harm; warnings about scams stay clean | 1:1 |
| Self-harm coercion (VN) | the **only 100%-model scenario**; the target's replies stay clean | 1:1 |
| Distress — must NOT blur (VN) | the **false positive**: flagged and labelled, never hidden | 1:1 |
| Group pile-on | the case the guardrail is **worst** at, with bystanders | group |
| Group: forwarded scam | intent is invisible — the member *warning* the group is flagged | group |
| Group: radicalisation (VN) | organising is invisible — **2 of 8** caught; region-invariant | group |
| Flood: 120 / 500 | load test; dumps an unscreened backlog, then scroll | group |

Watch with:

```bash
LOG_FILTER=1 ./scripts/run_sim.sh      # filters simulator noise; see the script header
```

Scripted, not generated, so verdicts are identical every run — which is what makes
them useful for spotting a regression. Each line declares `shouldFlag`, asserted by
`GuardrailDemoTests`.

> ⚠️ **Rules short-circuit**, so a scenario written to trip the UK patterns
> exercises the regex half *only* and would pass unchanged with the classifier
> completely broken. `testScenariosExerciseTheModelNotOnlyTheRules` forces at least
> one model-caught line. The same trap caught two test suites that were silently
> passing model-only because they never loaded rules — **always call
> `warmUp(language:rulesName:)` in tests and assert `ruleCount > 0`.**

---

## 11. Known limits

### The English model is weak

The single most important fact about this feature right now. Probing phrases that
match no rule:

```
Clean            0.911  "You know I could hurt you if I wanted to."
Clean            0.912  "Answer me right now or you'll be sorry."
Clean            0.910  "I hope something terrible happens to you."
Self-harm/Viol   0.839  "Come over tonight, no one has to know."
Self-harm/Viol   0.612  "You're worthless and everyone hates you."
```

Confident misses on plain English threats, and *self-harm* for two lines that are
nothing of the kind — consistent with `-en` being a copy of a Vietnamese-heavy
bilingual checkpoint.

**For the UK, the regex layer is doing nearly all the real work, and the model is
not a safety net behind it.** A genuine English pack is the highest-value
outstanding item.

### The detector is public

Both the weights and `guardrail-rules-uk-en.json` ship in the bundle — the rules as
readable text. Anyone can extract them and test offline until something passes.
This is inherent to on-device, not a flaw in the build; it sets what the feature
may claim. **Good against careless and impulsive harm, which is most of it.
Approximately nothing against a determined evader.**

### Pattern-level harm is invisible

A per-message classifier sees messages. Some of the most serious harms are not in
any single message:

- **Pile-on.** Four people being mildly unpleasant to one person, sustained. No
  individual message violates anything.
- **Intent.** The forwarded-scam scenario flags the member *warning* the group,
  identically to an author. The classifier has no notion of intent, which matters
  a great deal if per-user violation counts are ever built on top of it.

### Other gaps

- **Fail-open everywhere.** A broken load is indistinguishable from "all clean"
  except for the banner.
- **Classified once, never revisited**, with no record of what produced the
  verdict. Windowing shipped *after* messages had been classified without it, so
  those verdicts were computed on the first 128 tokens only and nothing will ever
  re-examine them. A **provenance stamp** (engine version, pack, rules region and
  version, threshold) would fix this and is the prerequisite for false-positive
  recovery and cross-device verdict sync.
- **No report, block, or mute** anywhere in the app.
- **Images are not screened.** `sendImage` never classifies. Image CSAM and
  cyberflashing are exactly what a text classifier cannot touch — that needs the
  image guardrail and, for CSAM, hash matching.
- **Group chat is not persisted at all.** It starts empty every launch, so verdicts
  dying with it is consistent rather than a leak — but there is no group history.
- **Group language fallback is weaker than 1:1.** A short message from a member
  writing in another language falls back to the *thread's* dominant language and is
  screened by the wrong pack. Falling back to that **author's** recent messages
  would be better.
- **Padding past 16 sentences escapes the model.** Rules are the backstop.
- **`uk-vi` has not had a native-speaker review**, and the false-positive corpus
  in `GuardUKRulesTests` is English-only. The 21 Vietnamese patterns were written
  against measured misses and follow the structural shape of Vietnamese scam
  messages, but idiom and register are unverified.
- **Only `political` is regional in `actionableClasses`.** Real policy differences
  are often *within* a class — Holocaust denial is criminal in Germany and lawful
  in the UK, but both are `hate` — and a class mask is too coarse to express that.
  Those differences have to live in patterns, which are language-locked.

---

## 12. Extending it

### Add a region (rules / policy)

1. Author `guardrails/regions/<region>-<lang>.yaml`. Categories must come from
   `ORDER`: `child, violence, self_harm, sexual, hate, harassment,
   illegal_goods, scam_fraud, political`.
2. `python3 scripts/build_regex_patterns.py --region <region>-<lang>`
3. Add the case in `GuardRegion.swift`: enum case, `displayName`, `regionCode`,
   **`ruleLanguages`** (or the file is never loaded) and **`actionableClasses`**.
   The Settings picker is `allCases`.
4. `xcodegen generate`
5. Update `testAuthoredFlagReflectsRealCoverage`.
6. **Write the region's benign corpus test.** Copy
   `testBenignUKEnglishIsNotFlagged` and write lines in that language that each aim
   at a specific rule of yours. Detection tests are easy; the false-positive list is
   what keeps the guardrail trusted.

Leave `political: []` unless the jurisdiction genuinely restricts political speech.
`\b` is useless for Chinese, Japanese and Thai — those need different anchoring.

### Add a language pack (model)

1. Drop `guardrail-bert-tiny-<lang>.tflite` and `guardrail-tokenizer-<lang>.json`
   into `Resources/`.
2. Add the `GuardPack` entry in `GuardPackCatalog.packs`.
3. `xcodegen generate`
4. Fix `testUnsupportedLanguageIsNotScreened` (it asserts French has no pack) and
   check `testCacheCanHoldTheWholeCatalog`.

No code change if the head shape matches — `decide` routes by length.

> **The tokenizer trap.** `GuardTokenizer` is a hand-port of *one specific* HF
> pipeline, validated against 70 fixtures. A checkpoint with a different normalizer
> or pre-tokenizer will load fine and tokenize **wrongly** — no error, just quietly
> degraded verdicts. Capture fresh fixtures and re-run the parity test.

---

## 13. Open items

Roughly in the order I would do them:

1. **A real English pack.** Everything in [§11](#11-known-limits) starts here.
2. **Provenance stamp.** Unblocks re-classification on upgrade, false-positive
   recovery, and cross-device verdict sync.
3. **Benchmark on a physical iPhone XS.** Every number in §9 is the simulator.
4. **Report / block.** The OSA expects it, and it is the input to any reputation
   design.
5. **Version-control `guardrails/`.** The UK policy file exists on one machine.
6. **False-positive run against a real English corpus.** The benign list in the
   tests is a smoke test, not a measurement.
7. Image guardrail · group collapse UI · per-author language fallback · adaptive
   compute budget (thermal, Low Power, backpressure) · native-speaker rule sets for
   France, Germany, Spain.

---

## Appendix

### File map

```
SLMChat/
  TFLiteC.h                       vendored TFLite C declarations
  GuardLM.h/.mm                   Obj-C++ bridge
  Engine/
    GuardTokenizer.swift          BPE port
    GuardRules.swift              regex layer
    GuardRegion.swift             jurisdictions
    GuardPackCatalog.swift        language → assets
    GuardEngine.swift             queue, cache, windowing, decision
    MessageScreener.swift         scheduling
    ChatService.swift             classify() + guardrailError
  Model/
    SafetyLabel.swift             verdict types
    GuardrailDemo.swift           scripted scenarios
    Conversation.swift            live + viewport paths (1:1)
    GroupChat.swift               live + viewport paths (group)
  View/
    MessageBubble.swift           blur, labels, accessibility
    ChatView.swift / GroupChatView.swift   viewport hooks, demo button
    SettingsView.swift            Safety section

Resources/guardrail-*.tflite|json  packs + compiled rules
guardrails/regions/*.yaml          rule SOURCES (gitignored)
scripts/build_regex_patterns.py    YAML → JSON
GuardrailTests/                    129 tests
```

### Log tags

| Tag | Meaning |
|---|---|
| `[Guard]` | one verdict, with classifier time |
| `[GuardRules]` | rule load count; which pattern matched |
| `[GuardEngine]` | pack load/evict; window count; work vs. queued |
| `[GuardPerf]` | main-actor tail over one frame |
| `[GuardDemo]` | scripted scenario progress, per-author tally, timings |
| `[GuardLM]` | TFLite tensor shapes at load |

### Glossary

| Term | Meaning |
|---|---|
| **Class** | what a message is — `SafetyLabel`, 10 values |
| **Severity** | how serious — `Severity`, 4 values; blur at Violation+ |
| **Region** | the reader's jurisdiction; selects the rules |
| **Pack** | one language's (model, tokenizer); selects the classifier |
| **Window** | one sentence, or the whole message; one forward pass each |
| **Authored** | a region contributed patterns of its own, beyond the universal baseline |
| **Live path** | screening on arrival |
| **Bulk / viewport path** | screening when a row scrolls into view |
| **Shedding** | dropping queued work far from the viewport; re-requested on return |
