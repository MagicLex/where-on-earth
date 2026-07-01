# Honesty notes

## What this is

A country-level classifier over frozen CLIP embeddings of street-view photos,
plus the baselines that keep it honest. It answers "which country does this look
like?" with a probability distribution over 200+ countries.

## What this is not

- **Not a geolocator.** The output is a country, the coarsest useful unit. No
  street, no coordinates, no address. The head is structurally incapable of more:
  it maps 512 CLIP floats to country logits, and CLIP embeddings famously discard
  exactly the fine detail (text on signs, house numbers) that doxxing would need.
- **Not a surveillance tool.** No face recognition, no person re-identification,
  no time inference. Uploaded photos are embedded, scored, and discarded; the app
  stores nothing.

## Why the numbers can be trusted

- **The split is by place.** Train and test are separated by OSV5M quadtree cell.
  Two frames of the same street, or the same drive, never sit on both sides. A
  row-level split would inflate top-1 dramatically and it would be a lie.
- **The baselines are strong on purpose.** Majority-class exposes class imbalance;
  CLIP zero-shot exposes how much of the skill was already in the backbone. The
  trained head is only worth its lift over the second one.
- **The label is ground truth, not a proxy.** OSV5M countries come from the GPS of
  the capture device. The rare mislabel (border zones) is noise, not bias.

## Known biases

- Mapillary coverage is dense in Europe, the US, and Brazil, thin in much of
  Africa and Central Asia. The per-country cap rebalances training, but rare
  countries still get scored on fewer held-out photos, so their per-country
  accuracy has wide error bars.
- Street view means roads. Deserts, interiors, food and faces are out of
  distribution; expect confident nonsense there and read the confidence bar, not
  just the flag.
