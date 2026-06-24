# Sample images

Curated demo images for the **Samples** tab. These are intentionally public static assets shipped with the app.

## Adding an image

1. Place your image file in this folder (e.g. `plate-daylight.jpg`).
2. Add an entry to `samples.json`:

```json
[
  {
    "id": "daylight-01",
    "src": "/samples/plate-daylight.jpg",
    "title": "Daylight plate",
    "expectedPlate": "ABC1234",
    "source": "own-vehicle",
    "consent": true
  }
]
```

3. Rebuild and deploy. Vercel will serve files from `public/samples/`.
4. Run `npm run validate:samples` from `frontend/yolo-plate-webApp` (catches missing files before CI).

### Current samples

| File | Title | Reference plate |
|------|-------|-----------------|
| `stock_ph_plates_1.png` | Clean plate — ABC 1230 | ABC 1230 |
| `stock_ph_plates_2.png` | Vintage PH plate — CJA 910 | CJA 910 |
| `stock_ph_plates_3.png` | Vintage PH plate — WPZ 225 | WPZ 225 |

`expectedPlate` is for demo reference only — OCR output may differ on weathered or stylized plates.

## Image policy

- Use **your own vehicle**, **synthetic** plates, or **licensed stock** only.
- Do **not** add random street, parking, or surveillance images with identifiable real plates without consent.
- Document `source` and `consent` in the manifest for your own records.

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique key for the gallery |
| `src` | yes | Path under `public/` (e.g. `/samples/foo.jpg`) |
| `title` | yes | Label shown in the gallery |
| `expectedPlate` | no | Reference text for demos (not validated automatically) |
| `source` | no | e.g. `own-vehicle`, `synthetic`, `stock` |
| `consent` | no | `true` if you have rights to publish the image |
