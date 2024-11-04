import Foundation
import whisper_cpp
import Accelerate

public class Whisper {
    private let whisperContext: OpaquePointer
    private var unmanagedSelf: Unmanaged<Whisper>?

    public var delegate: WhisperDelegate?
    public var params: WhisperParams
    public var newSegmentProbability: ((TokenSequence) -> Float)?
    public var completeSegmentProbability: ((TokenSequence) -> Float)?
    public private(set) var inProgress = false

    internal var frameCount: Int? // For progress calculation (value not in `whisper_state` yet)
    internal var cancelCallback: (() -> Void)?

    
    public var audioEnergy: [(rel: Float, avg: Float, max: Float, min: Float)] = []

    public init?(
        fromFileURL fileURL: URL,
        withParams params: WhisperParams = .default,
        newSegmentProbability: ((TokenSequence) -> Float)? = nil,
        completeSegmentProbability: ((TokenSequence) -> Float)? = nil)
    {
         guard let context = (fileURL.relativePath.withCString { whisper_init_from_file($0) }) else {
            return nil
        }
        self.whisperContext = context
        self.params = params
        self.newSegmentProbability = newSegmentProbability
        self.completeSegmentProbability = completeSegmentProbability
    }

    public init?(
        fromData data: Data,
        withParams params: WhisperParams = .default,
        newSegmentProbability: ((TokenSequence) -> Float)? = nil,
        completeSegmentProbability: ((TokenSequence) -> Float)? = nil)
    {
        var copy = data // Need to copy memory so we can gaurentee exclusive ownership over pointer

        guard let context = (copy.withUnsafeMutableBytes { 
            whisper_init_from_buffer($0.baseAddress!, data.count) }) else { return nil }

        self.whisperContext = context
        self.params = params
        self.newSegmentProbability = newSegmentProbability
        self.completeSegmentProbability = completeSegmentProbability
    }

    deinit {
        whisper_free(whisperContext)
    }

    private func prepareCallbacks() {
        /*
         C-style callbacks can't capture any references in swift, so we'll convert `self`
         to a pointer which whisper passes back as the `user_data` argument.

         We can unwrap that and obtain a copy of self inside the callback.
         */
        cleanupCallbacks()
        let unmanagedSelf = Unmanaged.passRetained(self)
        self.unmanagedSelf = unmanagedSelf
        params.new_segment_callback_user_data = unmanagedSelf.toOpaque()
        params.encoder_begin_callback_user_data = unmanagedSelf.toOpaque()
        params.progress_callback_user_data = unmanagedSelf.toOpaque()

        // swiftlint:disable line_length
        params.new_segment_callback = { (ctx: OpaquePointer?, _: OpaquePointer?, newSegmentCount: Int32, userData: UnsafeMutableRawPointer?) in
        // swiftlint:enable line_length
            guard let ctx = ctx,
                  let userData = userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()
            guard let delegate = whisper.delegate else { return }

            let segmentCount = whisper_full_n_segments(ctx)
            var newSegments: [Segment] = []
            newSegments.reserveCapacity(Int(newSegmentCount))

            let startIndex = segmentCount - newSegmentCount

            for index in startIndex..<segmentCount {
                guard let text = whisper_full_get_segment_text(ctx, index) else { continue }
                let startTime = whisper_full_get_segment_t0(ctx, index)
                let endTime = whisper_full_get_segment_t1(ctx, index)
                let probability: Float

                if let newSegmentProbability = whisper.newSegmentProbability {
                    probability = newSegmentProbability(TokenSequence(whisperContext: ctx, segmentIndex: index))
                } else {
                    probability = 1.0
                }

                newSegments.append(.init(
                    startTime: Int(startTime) * 10, // Time is given in ms/10, so correct for that
                    endTime: Int(endTime) * 10,
                    text: String(Substring(cString: text)),
                    probability: probability)
                )
            }

            DispatchQueue.main.async {
                delegate.whisper(whisper, didProcessNewSegments: newSegments, atIndex: Int(startIndex))
            }
        }

        params.encoder_begin_callback = { (_: OpaquePointer?, _: OpaquePointer?, userData: UnsafeMutableRawPointer?) in
            guard let userData = userData else { return true }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()

            if whisper.cancelCallback != nil {
                return false
            }

            return true
        }

        // swiftlint:disable line_length
        params.progress_callback = { (_: OpaquePointer?, _: OpaquePointer?, progress: Int32, userData: UnsafeMutableRawPointer?) in
        // swiftlint:enable line_length
            guard let userData = userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()

            DispatchQueue.main.async {
                whisper.delegate?.whisper(whisper, didUpdateProgress: Double(progress) / 100)
            }
        }
    }

    private func cleanupCallbacks() {
        guard let unmanagedSelf = unmanagedSelf else { return }

        unmanagedSelf.release()
        self.unmanagedSelf = nil
    }
    
    public func processBuffer(_ buffer: [Float]) {
//        audioSamples.append(contentsOf: buffer)

        // Find the lowest average energy of the last 20 buffers ~2 seconds
        let minAvgEnergy = self.audioEnergy.suffix(20).reduce(Float.infinity) { min($0, $1.avg) }
        let relativeEnergy = Self.calculateRelativeEnergy(of: buffer, relativeTo: minAvgEnergy)

        // Update energy for buffers with valid data
        let signalEnergy = Self.calculateEnergy(of: buffer)
        let newEnergy = (relativeEnergy, signalEnergy.avg, signalEnergy.max, signalEnergy.min)
        self.audioEnergy.append(newEnergy)
    }
    
    public static func calculateRelativeEnergy(of signal: [Float], relativeTo reference: Float?) -> Float {
        let signalEnergy = calculateAverageEnergy(of: signal)

        // Make sure reference is greater than 0
        // Default 1e-3 measured empirically in a silent room
        let referenceEnergy = max(1e-8, reference ?? 1e-3)

        // Convert to dB
        let dbEnergy = 20 * log10(signalEnergy)
        let refEnergy = 20 * log10(referenceEnergy)

        // Normalize based on reference
        // NOTE: since signalEnergy elements are floats from 0 to 1, max (full volume) is always 0dB
        let normalizedEnergy = rescale(value: dbEnergy, min: refEnergy, max: 0)

        // Clamp from 0 to 1
        return max(0, min(normalizedEnergy, 1))
    }
    
    public static func calculateAverageEnergy(of signal: [Float]) -> Float {
        var rmsEnergy: Float = 0.0
        vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))
        return rmsEnergy
    }
    
    public static func calculateEnergy(of signal: [Float]) -> (avg: Float, max: Float, min: Float) {
        var rmsEnergy: Float = 0.0
        var minEnergy: Float = 0.0
        var maxEnergy: Float = 0.0

        // Calculate the root mean square of the signal
        vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))

        // Calculate the maximum sample value of the signal
        vDSP_maxmgv(signal, 1, &maxEnergy, vDSP_Length(signal.count))

        // Calculate the minimum sample value of the signal
        vDSP_minmgv(signal, 1, &minEnergy, vDSP_Length(signal.count))

        return (rmsEnergy, maxEnergy, minEnergy)
    }


    public func transcribe(audioFrames: [Float], completionHandler: @escaping (Result<[Segment], Error>) -> Void) {
        prepareCallbacks()

        let wrappedCompletionHandler: (Result<[Segment], Error>) -> Void = { result in
            self.cleanupCallbacks()
            completionHandler(result)
        }

        guard !inProgress else {
            wrappedCompletionHandler(.failure(WhisperError.instanceBusy))
            return
        }
        guard audioFrames.count > 0 else {
            wrappedCompletionHandler(.failure(WhisperError.invalidFrames))
            return
        }

        inProgress = true
        frameCount = audioFrames.count
        
        processBuffer(audioFrames)
        
        printf("audioEnergy: \(self.audioEnergy)")

        DispatchQueue.global(qos: .userInitiated).async {
            whisper_full(self.whisperContext, self.params.whisperParams, audioFrames, Int32(audioFrames.count))

            let segmentCount = whisper_full_n_segments(self.whisperContext)

            var segments: [Segment] = []
            segments.reserveCapacity(Int(segmentCount))

            for index in 0..<segmentCount {
                guard let text = whisper_full_get_segment_text(self.whisperContext, index) else { continue }
                let startTime = whisper_full_get_segment_t0(self.whisperContext, index)
                let endTime = whisper_full_get_segment_t1(self.whisperContext, index)
                let probability: Float

                if let completeSegmentProbability = self.completeSegmentProbability {
                    probability = completeSegmentProbability(TokenSequence(whisperContext: self.whisperContext, segmentIndex: index))
                } else {
                    probability = 1.0
                }

                segments.append(.init(
                    startTime: Int(startTime) * 10, // Correct for ms/10
                    endTime: Int(endTime) * 10,
                    text: String(Substring(cString: text)),
                    probability: probability)
                )
            }

            if let cancelCallback = self.cancelCallback {
                DispatchQueue.main.async {
                    // Should cancel callback be called after delegate and completionHandler?
                    cancelCallback()

                    let error = WhisperError.cancelled

                    self.delegate?.whisper(self, didErrorWith: error)
                    wrappedCompletionHandler(.failure(error))
                }
            } else {
                DispatchQueue.main.async {
                    self.delegate?.whisper(self, didCompleteWithSegments: segments)
                    wrappedCompletionHandler(.success(segments))
                }
            }

            self.frameCount = nil
            self.cancelCallback = nil
            self.inProgress = false
        }
    }

    public func cancel(completionHandler: @escaping () -> Void) throws {
        guard inProgress else { throw WhisperError.cancellationError(.notInProgress) }
        guard cancelCallback == nil else { throw WhisperError.cancellationError(.pendingCancellation)}

        cancelCallback = completionHandler
    }

    @available(iOS 13, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
    public func transcribe(audioFrames: [Float]) async throws -> [Segment] {
        return try await withCheckedThrowingContinuation { cont in
            self.transcribe(audioFrames: audioFrames) { result in
                switch result {
                case .success(let segments):
                    cont.resume(returning: segments)
                case .failure(let error):
                    cont.resume(throwing: error)
                }
            }
        }
    }

    @available(iOS 13, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
    public func cancel() async throws {
        return try await withCheckedThrowingContinuation { cont in
            do {
                try self.cancel {
                    cont.resume()
                }
            } catch {
                cont.resume(throwing: error)
            }
        }
    }
}
