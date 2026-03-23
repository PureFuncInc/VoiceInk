import Foundation

struct TraditionalChineseConverter {
    /// 將文字中的簡體中文字元轉換為繁體中文（使用 Apple 內建 ICU transform Hans-Hant）
    static func convert(_ text: String) -> String {
        return text.applyingTransform(StringTransform("Hans-Hant"), reverse: false) ?? text
    }
}
