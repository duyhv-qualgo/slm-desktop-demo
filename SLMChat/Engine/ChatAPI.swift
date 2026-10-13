import Foundation

/// Cloud chat backend for Alex's replies, speaking Google's Gemini
/// **generateContent** API. The three enhancement features stay on the on-device
/// task model; only the chat partner uses this. Config is loaded from a gitignored
/// `APIConfig.plist` in the bundle — if that file is absent, `load()` returns nil
/// and the app falls back to the on-device chat model.
struct ChatAPIConfig {
    /// API base, e.g. `https://generativelanguage.googleapis.com/v1`.
    let baseURL: String
    let apiKey: String
    /// Gemini model id, e.g. `gemini-3.1-flash-lite`.
    let model: String

    /// Endpoint for a given Gemini method (`generateContent` /
    /// `streamGenerateContent`), keyed via the `x-goog-api-key` header (set in
    /// `ChatAPI`), so the key never lands in a URL.
    func endpoint(_ method: String, sse: Bool = false) -> URL? {
        let base = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        return URL(string: "\(base)/models/\(model):\(method)\(sse ? "?alt=sse" : "")")
    }

    static func load() -> ChatAPIConfig? {
        guard let url = ResourceLookup.url("APIConfig", ext: "plist"),
              let data = try? Data(contentsOf: url),
              let dict = try? PropertyListSerialization
                  .propertyList(from: data, format: nil) as? [String: Any],
              let base = dict["baseURL"] as? String, !base.isEmpty,
              let key = dict["apiKey"] as? String, !key.isEmpty else {
            return nil
        }
        let model = (dict["model"] as? String) ?? "gemini-3.1-flash-lite"
        return ChatAPIConfig(baseURL: base, apiKey: key, model: model)
    }
}

final class ChatAPI {
    private let config: ChatAPIConfig
    private let session: URLSession
    private(set) var lastError: String?

    init(config: ChatAPIConfig) {
        self.config = config
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 30
        self.session = URLSession(configuration: cfg)
    }

    // MARK: - Request building

    /// Map the windowed conversation to Gemini `contents`: `.you` → `user`,
    /// `.friend` → `model`. Consecutive same-role turns are merged (Gemini expects
    /// alternating roles) and empty-text turns dropped (empty parts are rejected).
    private func contents(history: [ChatMessage]) -> [[String: Any]] {
        var out: [[String: Any]] = []
        for m in history.suffix(Prompts.historyWindow) {
            let text = m.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let role = m.sender == .you ? "user" : "model"
            if var last = out.last, last["role"] as? String == role,
               var parts = last["parts"] as? [[String: String]] {
                parts.append(["text": text])
                last["parts"] = parts
                out[out.count - 1] = last
            } else {
                out.append(["role": role, "parts": [["text": text]]])
            }
        }
        return out
    }

    /// Assemble the Gemini request body: persona as `systemInstruction`, the mapped
    /// conversation as `contents`.
    private func body(persona: String, history: [ChatMessage]) -> [String: Any] {
        [
            "systemInstruction": ["parts": [["text": persona]]],
            "contents": contents(history: history),
            "generationConfig": [
                "temperature": 0.7,
                "maxOutputTokens": 512,
            ],
        ]
    }

    private func makeRequest(url: URL, body: [String: Any], sse: Bool) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(sse ? "text/event-stream" : "application/json",
                     forHTTPHeaderField: "accept")
        req.setValue(config.apiKey, forHTTPHeaderField: "x-goog-api-key")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        return req
    }

    /// Concatenate the text of every part in a Gemini candidate's content.
    private static func text(fromCandidates json: [String: Any]) -> String {
        guard let candidates = json["candidates"] as? [[String: Any]],
              let content = candidates.first?["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]] else { return "" }
        return parts.compactMap { $0["text"] as? String }.joined()
    }

    // MARK: - Streaming

    /// Streaming reply via `streamGenerateContent?alt=sse`. Emits each text chunk
    /// through `onChunk` as it arrives and returns the full assembled reply, or nil
    /// on error (`lastError`). `onChunk` may run off the main thread — callers must
    /// hop to the main actor before touching UI state.
    func streamReply(persona: String,
                     history: [ChatMessage],
                     onChunk: @escaping (String) -> Void) async -> String? {
        guard let url = config.endpoint("streamGenerateContent", sse: true) else {
            lastError = "Bad endpoint"; return nil
        }
        let req = makeRequest(url: url, body: body(persona: persona, history: history), sse: true)

        do {
            let (bytes, resp) = try await session.bytes(for: req)
            guard let http = resp as? HTTPURLResponse else {
                lastError = "No HTTP response"; return nil
            }
            guard (200..<300).contains(http.statusCode) else {
                var errText = ""
                for try await line in bytes.lines { errText += line }
                lastError = "HTTP \(http.statusCode): \(errText)"
                return nil
            }

            var full = ""
            // SSE frames are "data: {json}" lines; each JSON carries one incremental
            // candidate chunk.
            for try await line in bytes.lines {
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                guard let data = payload.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { continue }
                let piece = Self.text(fromCandidates: json)
                guard !piece.isEmpty else { continue }
                full += piece
                onChunk(piece)
            }

            let trimmed = full.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { lastError = "Empty stream"; return nil }
            return trimmed
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    // MARK: - Non-streaming

    /// Non-streaming reply via `generateContent`. Used by the burst seeder; the live
    /// chat uses `streamReply` for the typing effect. Returns the reply, or nil on
    /// error (`lastError`).
    func reply(persona: String, history: [ChatMessage]) async -> String? {
        guard let url = config.endpoint("generateContent") else {
            lastError = "Bad endpoint"; return nil
        }
        let req = makeRequest(url: url, body: body(persona: persona, history: history), sse: false)

        do {
            let (data, resp) = try await session.data(for: req)
            guard let http = resp as? HTTPURLResponse else {
                lastError = "No HTTP response"; return nil
            }
            guard (200..<300).contains(http.statusCode) else {
                lastError = "HTTP \(http.statusCode): \(String(data: data, encoding: .utf8) ?? "")"
                return nil
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                lastError = "Unexpected response shape"; return nil
            }
            let text = Self.text(fromCandidates: json).trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { lastError = "Empty response"; return nil }
            return text
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }
}
