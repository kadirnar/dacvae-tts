"""Usage: python examples/voice_clone.py checkpoint.pt reference.wav output.wav"""

import argparse

from dacvae_tts.inference import Synthesizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("ref_audio")
    parser.add_argument("output")
    parser.add_argument("--text", default="Hello! This sentence is generated using the reference voice.")
    args = parser.parse_args()
    tts = Synthesizer(args.checkpoint, device="cuda", asr_model="small.en")
    result = tts.synthesize(text=args.text, ref_audio=args.ref_audio, output=args.output)
    print(result.metadata)
    # For repeated requests, prepare once and reuse audio latents + inferred transcript:
    # voice = tts.prepare_reference(args.ref_audio)
    # tts.synthesize("Another sentence.", reference=voice, output="another.wav")


if __name__ == "__main__":
    main()
