import Foundation

/// Content class for a single message.
///
/// The raw values are a PERSISTENCE format, not the model's output indices. They
/// have meant the same thing since the 5-class SLM guardrail (0 clean, 1
/// violence, 2 sexual, 3 political, 4 child) and every stored message on every
/// device is encoded with them, so they must never be renumbered to match
/// whatever the current checkpoint happens to emit. `fromModelIndex` does that
/// translation instead — see it for the v16 mapping.
///
/// **Ten cases, eight of which the model can produce.** The set is deliberately
/// WIDER than the classifier's taxonomy, and the two extras are not leftovers:
///
///     model-backed  clean  violence  sexual  child  suicide  harassment
///                   illegal  scam
///     rules-only    political  hate
///
/// `political` and `hate` left the taxonomy in v16 and did not leave the
/// product. `vietnam-vi.yaml` carries 114 political patterns and `uk-en.yaml`
/// 22 hate patterns; a rule hit yields these labels exactly as before, and
/// `GuardReport` still offers both as report reasons.
///
/// That is the design, not a migration artefact. Whether political speech is
/// actionable is a jurisdiction question (`GuardRegion.actionableClasses`
/// subtracts it outside Vietnam), and stirring up racial or religious hatred is
/// an OSA Schedule 7 priority offence that the UK set must cover whether or not
/// a checkpoint happens to model it. Both are things a regex file can express
/// and audit line by line, which a model weight cannot.
///
/// Deleting either case does not raise an error — `GuardRules` logs the unknown
/// category and SKIPS it, so the patterns stay in the JSON and silently stop
/// mattering.
enum SafetyLabel: Int, Codable, CaseIterable {
    case clean = 0
    case violence = 1
    case sexual = 2
    case political = 3      // rules-only
    case child = 4
    case suicide = 5
    case hate = 6           // rules-only
    case harassment = 7
    case illegal = 8
    case scam = 9

    /// How many logits the current checkpoint's single head emits.
    ///
    /// Load-bearing: `GuardEngine.decide` identifies the class head by its width.
    /// The v15 model had two heads (10-class + 4-severity) and v16 has one of 8,
    /// so a stale value here does not throw — it silently matches no head and
    /// every message comes back Clean. Anything that changes this constant must
    /// change `fromModelIndex` with it.
    static let modelClassCount = 8

    /// Map a v16 output index to a label.
    ///
    /// v16 (`2026_08_25_v16_multilingual`) emits 8 classes and they are NOT this
    /// enum's raw values:
    ///
    ///     model 0 clean       → .clean       (raw 0)
    ///     model 1 violence    → .violence    (raw 1)
    ///     model 2 sexual      → .sexual      (raw 2)
    ///     model 3 child       → .child       (raw 4)
    ///     model 4 suicide     → .suicide     (raw 5)
    ///     model 5 harassment  → .harassment  (raw 7)
    ///     model 6 illegal     → .illegal     (raw 8)
    ///     model 7 scam        → .scam        (raw 9)
    ///
    /// The case NAMES match the model's class names exactly, so this table reads
    /// as an index shift and nothing else — the only thing that can be wrong here
    /// is a number. They used to be `childSafety`/`selfHarm`/`illegalGoods`/
    /// `scamFraud`, which meant every reading of this table was also a
    /// translation, and a confusion matrix from training could not be compared to
    /// a log line without one.
    ///
    /// Two labels have no model index at all — `political` (3) and `hate` (6) —
    /// which is why this is a table and not `SafetyLabel(rawValue:)`. Using the
    /// raw initialiser against v16 output would silently relabel every class from
    /// `child` upward: a scam would arrive as `illegal`, and child-safety
    /// content as `political`. Nothing would fail; the labels would just be wrong.
    static func fromModelIndex(_ index: Int) -> SafetyLabel? {
        switch index {
        case 0: return .clean
        case 1: return .violence
        case 2: return .sexual
        case 3: return .child
        case 4: return .suicide
        case 5: return .harassment
        case 6: return .illegal
        case 7: return .scam
        default: return nil
        }
    }

    var isUnsafe: Bool { self != .clean }

    /// What a PERSON reads — in the bubble, the inbox preview and the report
    /// sheet. Deliberately not the case name.
    ///
    /// The identifiers match the model so code and eval output line up; these
    /// don't, because the audience is different. "Suicide" as a chip under a
    /// message about self-cutting is both wrong and needlessly blunt — the
    /// category covers self-harm broadly, which is also what the rule files call
    /// it (`self_harm`, including the pro-ana patterns). "Illegal" alone says
    /// nothing; "Illegal goods" says what was detected.
    ///
    /// So three vocabularies meet in this file, on purpose:
    ///
    ///     model class   suicide        ← `fromModelIndex`, the eval matrix
    ///     Swift case    .suicide       ← matches the model, for debugging
    ///     YAML category self_harm      ← `GuardRules.categoryLabels`
    ///     display       "Self-harm"    ← what a user sees
    var displayName: String {
        switch self {
        case .clean:      return "Clean"
        case .violence:   return "Violence"
        case .sexual:     return "Sexual"
        case .political:  return "Political"
        case .child:      return "Child safety"
        case .suicide:    return "Self-harm"
        case .hate:       return "Hate"
        case .harassment: return "Harassment"
        case .illegal:    return "Illegal goods"
        case .scam:       return "Scam / fraud"
        }
    }
}

/// How serious a RULE considers a match. Read together with `SafetyLabel`: the
/// class says WHAT, the severity says HOW SERIOUS.
///
/// **Rules-only as of v16.** It used to be the model's second head (4-way), and
/// every verdict carried one. The v16 checkpoint is single-head — it emits a
/// class and nothing else — so the model can no longer say how serious a message
/// is, and `SafetyResult.severity` is now optional to say exactly that.
///
/// Deliberately NOT reconstructed from the class or from `confidence`. A
/// class→severity table would make severity a synonym for the class and stop
/// carrying information; a confidence→severity table would conflate "the model is
/// sure" with "this is dangerous", which are different axes — a crisply-worded
/// harassment message outranking a hedged child-safety one is exactly the
/// inversion that gets a blur decision wrong. Where severity is genuinely known
/// it comes from the pattern that matched, authored by a human in
/// `guardrails/regions/*.yaml`. Where it isn't, it is `nil` and the UI says
/// nothing rather than something invented.
enum Severity: Int, Codable, CaseIterable {
    case safe = 0
    case caution = 1
    case violation = 2
    case critical = 3

    var displayName: String {
        switch self {
        case .safe:      return "Safe"
        case .caution:   return "Caution"
        case .violation: return "Violation"
        case .critical:  return "Critical"
        }
    }
}

/// Which half of the guardrail produced a verdict.
///
/// The two halves fail in opposite directions and are fixed in different places,
/// so a log line that doesn't say which one spoke is close to unactionable. A bad
/// `rules` verdict is one line of YAML in `guardrails/regions/<region>-<lang>.yaml`
/// and can be fixed this afternoon; a bad `model` verdict needs a retrained pack.
/// The confidence hints at it — a rule is always 1.000 — but only by convention,
/// and a model genuinely can return 1.000.
enum SafetySource {
    /// A regex pattern matched. Carries the pattern text so a false positive
    /// traces to the exact line without grepping a second log statement.
    case rules(pattern: String)
    /// The BERT-Tiny classifier decided, at `confidence`.
    case model
    /// Nothing ran: empty text, no pack for the language, or the engine failed to
    /// load. **Distinct from a clean verdict** — this is the fail-open path, and
    /// counting it as "screened clean" is how coverage gaps stay invisible.
    case notScreened

    /// For the log line. Reads as `(FLAGGED by rule: <pattern>)`.
    var logDescription: String {
        switch self {
        case .rules:              return "by rule"
        case .model:              return "by model"
        case .notScreened:        return "not screened"
        }
    }
}

/// What one classification actually cost, split into the clocks that answer
/// different questions.
///
/// Wall time alone is not diagnosable. A message measured at 1172 ms had 16 ms of
/// `work` in it — the other 1156 ms was sitting on the guard queue behind other
/// messages, which is a scheduling problem, not a model problem, and the two are
/// fixed in completely different places. Reporting only the total sends you to
/// optimise an inference that was never the cost.
struct GuardTiming {
    /// Sat on the serial guard queue waiting its turn. Grows with `behind`.
    var waitMS: Double = 0
    /// Wall-clock time between being dequeued and producing a verdict.
    ///
    /// **Not the same as CPU spent** — see `cpuMS`. This is wall time on the guard
    /// queue, so it also counts a lazy pack load, the rules scan, and any period
    /// the thread was descheduled. A single forward pass measures 5–21 ms; when
    /// this reads a second, something other than arithmetic happened.
    var workMS: Double = 0

    /// CPU time this thread actually burned, from `CLOCK_THREAD_CPUTIME_ID`.
    ///
    /// The discriminator `workMS` alone cannot provide. The guard queue is
    /// `.utility`, so under load it gets descheduled rather than run slower, and
    /// wall time inflates while the work is unchanged:
    ///
    ///     1042.2work / 12.4cpu   → starved; the model was never the cost
    ///     1042.2work / 1039.1cpu → genuinely computing for a second
    ///
    /// Valid because the whole body runs on one thread of the serial queue.
    var cpuMS: Double = 0
    /// Forward passes run. Every window is a full pass over a padded 128 tokens,
    /// so this is the multiplier on `workMS` — a 2-sentence message is 3 windows.
    var windows: Int = 0
    /// Messages still queued after this one finished. The number a flood control
    /// would act on.
    var behind: Int = 0

    var totalMS: Double { waitMS + workMS }
}

/// Verdict for one classified message.
struct SafetyResult {
    let label: SafetyLabel

    /// How serious, when that is actually known — i.e. when a RULE decided.
    ///
    /// `nil` for every model verdict since v16, which has no severity head. It is
    /// optional rather than defaulted to `.safe` because those are opposite
    /// claims: `.safe` asserts the content is fine, `nil` says nobody assessed it.
    /// Defaulting would have made every model-flagged message look benign to
    /// anything reading severity, including the blur gate.
    let severity: Severity?

    /// For a model verdict this is `p_unsafe` — `1 − softmax(logits)[clean]`, the
    /// total probability mass on unsafe classes, NOT the winning class's own
    /// probability. v16's decision rule sums that mass before comparing to the
    /// threshold, so a message split evenly across `violence` and `harassment`
    /// can flag even though neither class dominates. A rule hit is always 1.
    let confidence: Float

    /// Defaulted to `.model` so the many places that build a verdict by hand —
    /// tests, fixtures — keep compiling and keep meaning what they meant.
    var source: SafetySource = .model

    /// Stamped by `GuardEngine` on the way out. Diagnostic only — nothing about
    /// the verdict depends on it, and it is not persisted.
    var timing = GuardTiming()

    var isUnsafe: Bool { label.isUnsafe }

    /// The benign default, also used as a fail-open result when the model can't run.
    static let clean = SafetyResult(label: .clean, severity: .safe,
                                    confidence: 1, source: .notScreened)
}
