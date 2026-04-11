"""
Arabic MSA Noise Injector — Final Version
AraSpell method, 0.12 distortion ratio, clean output.
Output: noisy_data.json with only Safe/Unsafe, Category, Prompt fields.
"""
import json, random, re, argparse

ARABIC_CHARS = list('ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآءةىؤئ')
AR = re.compile(r'[\u0600-\u06FF]')

KEYBOARD = {
    'ض':['ص','ط'],'ص':['ض','ث','ق'],'ث':['ص','ق','ف'],'ق':['ث','ف','غ'],
    'ف':['ق','غ','ع'],'غ':['ف','ع','ه'],'ع':['غ','ه','خ'],'ه':['ع','خ','ح'],
    'خ':['ه','ح','ج'],'ح':['خ','ج'],'ج':['ح'],'ش':['س','ي'],'س':['ش','ن','ب'],
    'ي':['ش','ب','ل'],'ب':['ي','ل','ا'],'ل':['ب','ا','ك'],'ا':['ل','ك','ت'],
    'ك':['ل','ت','م'],'ت':['ك','م','ن'],'م':['ك','ن'],'ن':['م','ت'],
    'ظ':['ط','ذ'],'ط':['ظ','ذ','ض'],'ذ':['ط','ظ','د'],'د':['ذ','ز'],
    'ز':['د','ر'],'ر':['ز','و'],'و':['ر'],
}

MAPPINGS = {
    'أ':['ا','إ'],'إ':['ا','أ'],'آ':['ا'],'ء':['ا','أ'],'ئ':['ي'],'ؤ':['و'],
    'ة':['ه'],'ه':['ة'],
    'ى':['ي'],'ي':['ى'],
    'ض':['ظ'],'ظ':['ض'],
    'ذ':['د'],'ث':['س'],
    'ش':['س'],'خ':['ح'],'غ':['ع'],'ق':['ف'],'ز':['ر'],
}

def distort(text: str, n_ops: int) -> str:
    chars = list(text)
    for _ in range(n_ops):
        ar_pos = [i for i,c in enumerate(chars) if AR.match(c)]
        if not ar_pos:
            break
        pos = random.choice(ar_pos)
        ch = chars[pos]
        ops = []
        if ch in MAPPINGS:
            ops += ['map'] * 4
        ops += ['sub'] * 2
        ops += ['transpose'] * 2
        ops += ['insert']
        ops += ['delete']
        op = random.choice(ops)
        if op == 'map':
            chars[pos] = random.choice(MAPPINGS[ch])
        elif op == 'sub':
            if ch in KEYBOARD and random.random() < 0.7:
                chars[pos] = random.choice(KEYBOARD[ch])
            else:
                chars[pos] = random.choice(ARABIC_CHARS)
        elif op == 'transpose':
            if pos < len(chars) - 1:
                chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            elif pos > 0:
                chars[pos], chars[pos-1] = chars[pos-1], chars[pos]
        elif op == 'insert':
            chars.insert(pos, random.choice(ARABIC_CHARS))
        elif op == 'delete':
            chars.pop(pos)
    return ''.join(chars)

def apply_noise(text: str, distortion_ratio: float = 0.12) -> str:
    arabic_count = sum(1 for c in text if AR.match(c))
    n_ops = max(1, round(arabic_count * distortion_ratio))
    return distort(text, n_ops)

def process_json(input_path, output_path, distortion_ratio, seed=None):
    if seed is not None:
        random.seed(seed)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []; changed = 0
    for entry in data:
        if 'Prompt' not in entry or not entry['Prompt']:
            continue
        noisy_prompt = apply_noise(entry['Prompt'], distortion_ratio)
        # Clean output — only these 3 fields
        ne = {
            'Safe/Unsafe': entry.get('Safe/Unsafe', ''),
            'Category':    entry.get('Category', ''),
            'Prompt':      noisy_prompt,
        }
        if noisy_prompt != entry['Prompt']:
            changed += 1
        out.append(ne)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✓ Done.")
    print(f"  Entries   : {len(out)}")
    print(f"  Changed   : {changed}")
    print(f"  Distortion: {distortion_ratio}")
    print(f"  Output    : {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--distortion', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)
    process_json(args.input, 'noisy_data.json', args.distortion, args.seed)
