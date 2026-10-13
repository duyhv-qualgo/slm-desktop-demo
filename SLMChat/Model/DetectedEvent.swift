import Foundation

/// Structured event pulled out of a chat message by the on-device task finetune's
/// `event_creation` task. All fields are strings (empty when the model couldn't
/// find that field), matching the model's JSON contract.
///
/// Raw model output is noisy — interrogative fillers ("mấy giờ" / "what time"),
/// relative dates ("mai" / "tomorrow"), preposition prefixes ("ở lớp"). Decoding
/// runs `cleaned()` so callers see de-noised fields, and `resolvedStart` /
/// `displayDate` / `displayTime` surface actionable absolute values.
struct DetectedEvent: Codable, Equatable {
    var intent: String
    var eventName: String
    var date: String
    var time: String
    var location: String

    private enum CodingKeys: String, CodingKey {
        case intent
        case eventName = "event_name"
        case date
        case time
        case location
    }

    init(intent: String = "", eventName: String = "", date: String = "",
         time: String = "", location: String = "") {
        self.intent = intent
        self.eventName = eventName
        self.date = date
        self.time = time
        self.location = location
    }

    /// Tolerant decode: the model may omit fields entirely, so every key is
    /// optional here and defaults to "". Runs cleanup so junk never reaches the UI.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        intent = (try? c.decode(String.self, forKey: .intent)) ?? ""
        eventName = (try? c.decode(String.self, forKey: .eventName)) ?? ""
        date = (try? c.decode(String.self, forKey: .date)) ?? ""
        time = (try? c.decode(String.self, forKey: .time)) ?? ""
        location = (try? c.decode(String.self, forKey: .location)) ?? ""
        self = cleaned()
    }

    /// True when *any* event field survived cleanup. A loose gate — the model fires
    /// `has_event` liberally, so use `isActionable` to decide whether to notify.
    var hasEvent: Bool { !eventName.isEmpty || !date.isEmpty || !time.isEmpty }

    /// Whether the event is concrete enough to be worth surfacing a reminder for.
    /// The model flags farewell pleasantries — "hẹn gặp lại" / "see you tomorrow" —
    /// as events, which are false alarms. We score concreteness and require a modest
    /// total so a name+vague-day pleasantry (e.g. "gặp lại" + "sáng mai", no time,
    /// no place) stays silent, while a real plan (a clock time, or a place, or a
    /// meaningful title with a day) surfaces.
    var isActionable: Bool {
        var score = 0
        if parsedTime != nil { score += 2 }                            // a clock time is the strongest signal
        if !date.isEmpty { score += 1 }                                // a day (resolvable or explicit)
        if !location.isEmpty { score += 1 }                            // a place
        if !eventName.isEmpty && !Self.isWeakName(eventName) { score += 1 }  // a real title, not "see you"
        return score >= 2
    }

    /// Titles that are farewell/greeting idioms or bare "meet" verbs — on their own
    /// they signal a sign-off, not a plan, so they don't count toward `isActionable`.
    private static let weakNames: Set<String> = [
        "gặp", "gặp lại", "gặp nhau", "hẹn gặp", "hẹn gặp lại", "gặp lại sau",
        "tạm biệt", "chào", "chào nhé", "bye", "goodbye",
        "see you", "see you again", "see you later", "see u", "cya",
    ]

    private static func isWeakName(_ name: String) -> Bool {
        weakNames.contains(name.lowercased().trimmingCharacters(in: .whitespaces))
    }

    // MARK: - Display

    /// Human date, preferring a resolved absolute date ("Tomorrow · Thu, Jul 23")
    /// and falling back to the model's raw string when we can't resolve it.
    var displayDate: String? {
        if let d = resolvedStart {
            let df = DateFormatter()
            df.dateFormat = "EEE, MMM d"
            let base = df.string(from: d)
            if let rel = Self.relativeLabel(for: d) { return "\(rel) · \(base)" }
            return base
        }
        return date.isEmpty ? nil : date
    }

    /// Normalized clock time ("7:15 AM"), falling back to the raw string.
    var displayTime: String? {
        if let (h, m) = parsedTime {
            var comps = DateComponents()
            comps.hour = h; comps.minute = m
            if let d = Calendar.current.date(from: comps) {
                let df = DateFormatter()
                df.dateFormat = "h:mm a"
                return df.string(from: d)
            }
        }
        return time.isEmpty ? nil : time
    }

    /// "Tomorrow · Thu, Jul 23 · 7:15 AM · lớp" from whichever parts exist.
    var subtitle: String {
        [displayDate, displayTime, location.isEmpty ? nil : location]
            .compactMap { $0 }
            .joined(separator: " · ")
    }

    /// Absolute start `Date` combining the resolved date and parsed time — the
    /// value a "Add to Calendar" / reminder action should schedule against.
    /// Returns nil when we can't pin down at least a date.
    var resolvedStart: Date? {
        guard let day = resolvedDay else { return nil }
        guard let (h, m) = parsedTime else { return day }
        return Calendar.current.date(bySettingHour: h, minute: m, second: 0, of: day)
    }

    // MARK: - Post-processing

    /// Trim fields, strip preposition prefixes, and null out interrogative /
    /// placeholder junk ("mấy giờ", "what time", "?", "n/a", …).
    func cleaned() -> DetectedEvent {
        DetectedEvent(
            intent: intent.trimmingCharacters(in: .whitespacesAndNewlines),
            eventName: Self.scrub(eventName),
            date: Self.scrub(date),
            time: Self.scrub(time),
            location: Self.stripLeadingPreposition(Self.scrub(location))
        )
    }

    /// Values (lowercased) that are questions or placeholders, not real data.
    private static let junkValues: Set<String> = [
        "mấy giờ", "mấy h", "mấy", "bao giờ", "khi nào", "lúc nào", "chưa",
        "chưa rõ", "chưa biết", "không rõ", "không biết", "không",
        "what time", "when", "unknown", "n/a", "na", "none", "tbd", "?", "-",
    ]

    private static func scrub(_ raw: String) -> String {
        let v = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = v.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "?."))
        if v.isEmpty || junkValues.contains(v.lowercased()) || junkValues.contains(key) {
            return ""
        }
        return v
    }

    /// Drop a leading locative preposition so "ở lớp" / "at room A" → "lớp" / "room A".
    private static func stripLeadingPreposition(_ v: String) -> String {
        let prefixes = ["ở ", "tại ", "at ", "in "]
        for p in prefixes where v.lowercased().hasPrefix(p) {
            return String(v.dropFirst(p.count)).trimmingCharacters(in: .whitespaces)
        }
        return v
    }

    // MARK: - Relative date / time resolution

    /// Resolve the `date` string to an absolute day (start-of-day), handling
    /// Vietnamese + English relative terms and weekday names. nil if unrecognized.
    private var resolvedDay: Date? {
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        let d = date.lowercased()
        guard !d.isEmpty else { return nil }

        func addingDays(_ n: Int) -> Date? { cal.date(byAdding: .day, value: n, to: today) }

        if d.contains("hôm nay") || d == "nay" || d.contains("today") { return today }
        if d.contains("ngày mai") || d == "mai" || d.contains("tomorrow") { return addingDays(1) }
        if d.contains("ngày mốt") || d == "mốt" || d.contains("day after") { return addingDays(2) }
        if d.contains("cuối tuần") || d.contains("weekend") { return Self.nextWeekday(7, from: today) } // Saturday

        // Weekday names → next occurrence (VN thứ 2..thứ 7 = Mon..Sat, chủ nhật = Sun).
        let weekdayMap: [(keys: [String], weekday: Int)] = [
            (["chủ nhật", "cn", "sunday"], 1),
            (["thứ hai", "thứ 2", "monday"], 2),
            (["thứ ba", "thứ 3", "tuesday"], 3),
            (["thứ tư", "thứ 4", "wednesday"], 4),
            (["thứ năm", "thứ 5", "thursday"], 5),
            (["thứ sáu", "thứ 6", "friday"], 6),
            (["thứ bảy", "thứ 7", "saturday"], 7),
        ]
        for entry in weekdayMap where entry.keys.contains(where: d.contains) {
            return Self.nextWeekday(entry.weekday, from: today)
        }
        return nil
    }

    /// Next date (including today) whose weekday matches `target` (1 = Sunday).
    private static func nextWeekday(_ target: Int, from day: Date) -> Date? {
        let cal = Calendar.current
        for offset in 0..<7 {
            if let cand = cal.date(byAdding: .day, value: offset, to: day),
               cal.component(.weekday, from: cand) == target {
                return cand
            }
        }
        return nil
    }

    private static func relativeLabel(for date: Date) -> String? {
        let cal = Calendar.current
        if cal.isDateInToday(date) { return "Today" }
        if cal.isDateInTomorrow(date) { return "Tomorrow" }
        return nil
    }

    /// Parse the `time` string to (hour24, minute), tolerating "7h15", "7h",
    /// "7:15", "8 giờ 30", "Tầm 7h15", "2 pm". nil if no clock time is present.
    private var parsedTime: (Int, Int)? {
        let t = time.lowercased()
        guard let m = t.range(of: #"(\d{1,2})\s*(?:h|:|giờ|g)\s*(\d{1,2})?|(\d{1,2})\s*(am|pm)"#,
                              options: .regularExpression) else { return nil }
        let match = String(t[m])
        let nums = match.components(separatedBy: CharacterSet.decimalDigits.inverted)
            .compactMap { Int($0) }
        guard let first = nums.first else { return nil }
        var hour = first
        let minute = nums.count > 1 ? nums[1] : 0
        if match.contains("pm"), hour < 12 { hour += 12 }
        if match.contains("am"), hour == 12 { hour = 0 }
        guard (0...23).contains(hour), (0...59).contains(minute) else { return nil }
        return (hour, minute)
    }

    // MARK: - Parsing

    /// Parse the model's raw output into a `DetectedEvent`. Tolerates ```json
    /// fences / surrounding prose by extracting the first balanced `{ … }` block.
    static func parse(_ raw: String) -> DetectedEvent? {
        guard let jsonString = firstBraceBlock(in: raw),
              let data = jsonString.data(using: .utf8) else {
            return nil
        }
        return try? JSONDecoder().decode(DetectedEvent.self, from: data)
    }

    /// Return the substring from the first `{` to its matching `}` (brace-balanced),
    /// or nil if none is found.
    private static func firstBraceBlock(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var i = start
        while i < text.endIndex {
            let ch = text[i]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 {
                    return String(text[start...i])
                }
            }
            i = text.index(after: i)
        }
        return nil
    }
}
