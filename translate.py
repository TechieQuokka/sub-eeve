#!/usr/bin/env python3
"""
EEVE-Korean Subtitle Translator CLI

Usage:
    python translate.py --input subtitle.json --output subtitle_kr.json \\
                        --input-lang ja --target-lang ko
"""
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_model, translate_segment, translate_batch, load_json, save_json, get_field_name
from config import LANGUAGE_MAP, CHUNK_SIZE, CONTEXT_BEFORE, CONTEXT_AFTER, SHOW_PROGRESS, BEST_OF_N


def main():
    parser = argparse.ArgumentParser(
        description='EEVE-Korean Subtitle Translator (Q8_0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Japanese to Korean (recommended)
  python translate.py --input subtitle.json --output subtitle_kr.json --input-lang ja --target-lang ko

  # English to Korean
  python translate.py --input subtitle.json --output subtitle_kr.json --input-lang en --target-lang ko

  # With batch mode (faster, better context)
  python translate.py --input subtitle.json --output subtitle_kr.json --input-lang ja --target-lang ko --batch

  # Without context (faster but less coherent)
  python translate.py --input subtitle.json --output subtitle_kr.json --input-lang ja --target-lang ko --no-context

Supported languages: ja (Japanese), en (English), ko (Korean), zh (Chinese)
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON subtitle file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file')
    parser.add_argument('--input-lang', required=True,
                       choices=list(LANGUAGE_MAP.keys()),
                       help='Source language code')
    parser.add_argument('--target-lang', required=True,
                       choices=list(LANGUAGE_MAP.keys()),
                       help='Target language code (usually ko)')

    # Optional arguments
    parser.add_argument('--batch', action='store_true',
                       help='Use batch translation mode (translate multiple segments together)')
    parser.add_argument('--batch-size', type=int, default=CHUNK_SIZE,
                       help=f'Number of segments per batch (default: {CHUNK_SIZE})')
    parser.add_argument('--no-context', action='store_true',
                       help='Disable context-aware translation')
    parser.add_argument('--model', type=str,
                       help='Custom model path (overrides config.py)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--best-of', type=int, default=BEST_OF_N,
                       help=f'Generate N candidates and select best (1=disabled, 3-5 recommended, default: {BEST_OF_N})')

    args = parser.parse_args()

    print("=" * 70)
    print("🌟 EEVE-Korean Subtitle Translator (Q8_0)")
    print("=" * 70)

    # Step 1: Load model
    print("\n[1/4] Loading EEVE-Korean model...")
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}", file=sys.stderr)
        print("\nPlease download the model:")
        print("  huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \\")
        print("    --include '*Q8_0.gguf' --local-dir ../models")
        sys.exit(1)

    # Step 2: Load input JSON
    print(f"\n[2/4] Loading input file: {args.input}")
    try:
        data = load_json(args.input)
        segments = data['segments']
        print(f"  ✓ Loaded {len(segments)} segments")
    except Exception as e:
        print(f"\n❌ Failed to load input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Translate segments
    use_context = not args.no_context
    use_batch = args.batch
    best_of = args.best_of
    mode_desc = []
    if use_batch:
        mode_desc.append(f"batch mode (size={args.batch_size})")
    if use_context:
        mode_desc.append(f"with context (before={CONTEXT_BEFORE}, after={CONTEXT_AFTER})")
    if best_of > 1:
        mode_desc.append(f"best-of-{best_of} sampling")
    mode_str = ", ".join(mode_desc) if mode_desc else "single-segment mode"

    print(f"\n[3/4] Translating {LANGUAGE_MAP[args.input_lang]} → {LANGUAGE_MAP[args.target_lang]}")
    print(f"  Mode: {mode_str}")

    field_name = get_field_name(args.target_lang)
    failed_count = 0
    show_progress = SHOW_PROGRESS and not args.no_progress

    try:
        if use_batch:
            # Batch translation mode
            total_batches = (len(segments) + args.batch_size - 1) // args.batch_size
            iterator = range(0, len(segments), args.batch_size)

            if show_progress:
                iterator = tqdm(iterator, total=total_batches, desc="Translating", unit="batch")

            for batch_start in iterator:
                batch_end = min(batch_start + args.batch_size, len(segments))
                batch_segments = segments[batch_start:batch_end]

                # Build bidirectional context
                context_before = None
                context_after = None
                if use_context:
                    # Previous context
                    if batch_start > 0:
                        ctx_start = max(0, batch_start - CONTEXT_BEFORE)
                        context_before = [segments[j]['text'] for j in range(ctx_start, batch_start)]

                    # Following context
                    if batch_end < len(segments):
                        ctx_end = min(len(segments), batch_end + CONTEXT_AFTER)
                        context_after = [segments[j]['text'] for j in range(batch_end, ctx_end)]

                # Translate batch
                batch_texts = [seg['text'] for seg in batch_segments]
                translations = translate_batch(
                    batch_texts,
                    args.input_lang,
                    args.target_lang,
                    model,
                    context_before,
                    context_after,
                    best_of
                )

                # Update segments
                for i, translation in enumerate(translations):
                    segments[batch_start + i][field_name] = translation
                    if translation.startswith("[") and "Error" in translation:
                        failed_count += 1

        else:
            # Single-segment mode
            iterator = enumerate(segments)

            if show_progress:
                iterator = tqdm(iterator, total=len(segments), desc="Translating", unit="segment")

            for i, seg in iterator:
                # Build bidirectional context
                context_before = None
                context_after = None
                if use_context:
                    # Previous context
                    if i > 0:
                        ctx_start = max(0, i - CONTEXT_BEFORE)
                        context_before = [segments[j]['text'] for j in range(ctx_start, i)]

                    # Following context
                    if i < len(segments) - 1:
                        ctx_end = min(len(segments), i + 1 + CONTEXT_AFTER)
                        context_after = [segments[j]['text'] for j in range(i + 1, ctx_end)]

                # Translate
                translation = translate_segment(
                    seg['text'],
                    args.input_lang,
                    args.target_lang,
                    model,
                    context_before,
                    context_after,
                    best_of
                )
                seg[field_name] = translation

                # Track failures
                if translation.startswith("[") and "Error" in translation:
                    failed_count += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Translation interrupted by user")
        print("Saving progress...")

    except Exception as e:
        print(f"\n❌ Translation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 4: Save results
    print(f"\n[4/4] Saving results to: {args.output}")

    # Remove existing file if it exists
    output_path = Path(args.output)
    if output_path.exists():
        print(f"  ⚠️  Existing file found, removing...")
        try:
            output_path.unlink()
            print(f"  ✓ Existing file removed")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not remove existing file: {e}")

    try:
        save_json(data, args.output)
        print(f"  ✓ Saved successfully")
    except Exception as e:
        print(f"\n❌ Failed to save output: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("✅ Translation Complete!")
    print("=" * 70)
    print(f"Total segments:  {len(segments)}")
    print(f"Successful:      {len(segments) - failed_count}")
    if failed_count > 0:
        print(f"Failed:          {failed_count}")
    print(f"Output file:     {args.output}")
    print(f"Translation:     {LANGUAGE_MAP[args.input_lang]} → {LANGUAGE_MAP[args.target_lang]}")
    print("=" * 70)


if __name__ == '__main__':
    main()
