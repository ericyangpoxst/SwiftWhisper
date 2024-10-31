import Foundation
import whisper_cpp

public struct Token: Equatable {
    public let text: String
    public let probability: Float
}

public struct TokenSequence: Sequence, IteratorProtocol {
    public typealias Element = Token

    private let whisperContext: OpaquePointer
    private let segmentIndex: Int32
    private var tokenIndex: Int32
    private var numTokens: Int32

    init(whisperContext: OpaquePointer, segmentIndex: Int32) {
        self.whisperContext = whisperContext
        self.segmentIndex = segmentIndex
        self.tokenIndex = 0
        self.numTokens = -1
    }

    mutating public func next() -> Token? {
        if numTokens == -1 {
            numTokens = whisper_full_n_tokens(whisperContext, segmentIndex)
        }
        guard tokenIndex < numTokens else {
            return nil
        }
        guard let tokenText = whisper_full_get_token_text(whisperContext, segmentIndex, tokenIndex) else {
            return nil
        }
        let probability = whisper_full_get_token_p(whisperContext, segmentIndex, tokenIndex)
        tokenIndex += 1
        return Token(text: String(Substring(cString: tokenText)), probability: probability)
    }
}
