# Audio Library Loader Node

The voice node will:

* Accept a folder path which contains audio file assets.
* Scan the folder and once scanned, a cache is saved in a JSON file. 
* The cache can be refreshed if you click on re-scan folder.
* File names should be `Audio_01.wav` & `Audio_01.wav.txt` for the audio file and the reference text that accompanies it. Any other naming convention will be ignored.
* The selections will be populated via javascript. And passed into the `FG_TTSSpeak` node (Fish S2 Pro or Higgs v3).

The node once finished scanning will look like this:
```
  ┌──────────────────────────────────────────┐
  │           Audio Library Loader        ⟳ │ <-- Refresh to refresh JSON cache javascript trigger
  ├──────────────────────────────────────────┤
  │ ○ JSON Dialogue      Audio Selection  ○  │
  │                                          │
  │ <→Input _folder_path>                    │
  │                                          │
  │   Audio_01.wav | Audio_01.wav.txt        │ <-- Auto populated by scanned file
  │ ✓ Audio_02.wav | Audio_02.wav.txt        │ <-- Can be auto-selected by user or passed in JSON file
  │ ✓ Audio_03.wav | Audio_03.wav.txt        │
  │                                          │
  └──────────────────────────────────────────┘
```
* The `JSON` schema should be passed in like so:

  ```JSON
  {
    "dialogue_01.flac":{
      "audio_ref":"Audio_01.wav",
      "dialogue":"Hello there!"
    },
    "dialogue_02.flac":{
      "audio_ref":"Audio_02.wav",
      "dialogue":"General Kenobi!"
    },
    "dialogue_03.flac":{
      "audio_ref":"Audio_01.wav",
      "dialogue":"I've been waiting for you."
    }
  }
  ```

