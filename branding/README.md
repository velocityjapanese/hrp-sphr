# HarpSpheric channel branding

Channel: **HarpSpheric** — `@harpspheric` — `UCt3A4vWGA9q50xH3hg-9JLQ`

| Asset | File | Status |
| --- | --- | --- |
| Banner | `channel_banner_2048x1152.jpg` | set through the API |
| Description | `channel_description.txt` | set through the API |
| Keywords | `channel_keywords.txt` | set through the API |
| Avatar | `channel_avatar_800.png` | **manual upload only** |
| Video watermark | not used | the sheet-music panel owns the upper-right corner |

## What the YouTube Data API can set

`channels.update` accepts only the `brandingSettings`, `invideoPromotion` and
`localizations` parts. For branding it can change the channel description,
keywords, country, default language, trailer and banner.

The banner is a two-step flow:

1. `channelBanners.insert` uploads the image (16:9, at least 2048x1152).
2. `channels.update` stores the returned URL in
   `brandingSettings.image.bannerExternalUrl`.

Run the helper:

```powershell
python scripts/update_channel_branding.py `
  --token-json "D:\E agy cli\E api tokens\.HarpSpheric\token_HarpSpheric.json" `
  --confirm-channel HarpSpheric --apply
```

## What it cannot set

- **Channel avatar.** There is no `channels.avatar` method and no avatar field
  in `brandingSettings.image` (that object only carries banner artwork).
  Upload `channel_avatar_800.png` in YouTube Studio → Customisation → Branding.
- **Channel title.** `channels.update` returns
  `channelTitleUpdateForbidden` if the title changes. The channel is already
  named HarpSpheric, so nothing needs to be done.
- **Handle / custom URL and links.** These are Studio-only settings.

These limits are from the current `channels.update` reference:
<https://developers.google.com/youtube/v3/docs/channels/update>
