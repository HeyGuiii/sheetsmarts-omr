"""
SheetSmarts OMR Service
Converts photos of sheet music to structured JSON using homr (Optical Music Recognition).
"""

import base64
import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="SheetSmarts OMR")

# Allow CORS from the SheetSmarts frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str  # base64 encoded image


# Duration mapping: MusicXML type to Tone.js notation
DURATION_MAP = {
    "whole": "1n",
    "half": "2n",
    "quarter": "4n",
    "eighth": "8n",
    "16th": "16n",
    "32nd": "32n",
}


def musicxml_to_score(xml_string: str) -> dict:
    """Convert MusicXML to our internal score JSON format."""
    root = ET.fromstring(xml_string)

    # Namespace handling — MusicXML may or may not have namespaces
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    score = {
        "title": None,
        "timeSignature": [4, 4],
        "keySignature": "C",
        "tempo": 100,
        "notes": [],
    }

    # Try to get title
    title_el = root.find(f".//{ns}work-title")
    if title_el is not None and title_el.text:
        score["title"] = title_el.text
    else:
        title_el = root.find(f".//{ns}movement-title")
        if title_el is not None and title_el.text:
            score["title"] = title_el.text

    # Key signature map (fifths to key name)
    fifths_to_key = {
        -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
        0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
    }

    # Process each part
    parts = root.findall(f".//{ns}part")
    for part_idx, part in enumerate(parts):
        hand = "right" if part_idx == 0 else "left"
        divisions = 1  # default

        for measure in part.findall(f"{ns}measure"):
            measure_num = int(measure.get("number", 1))

            # Check for attributes (time sig, key sig, divisions)
            attributes = measure.find(f"{ns}attributes")
            if attributes is not None:
                div_el = attributes.find(f"{ns}divisions")
                if div_el is not None:
                    divisions = int(div_el.text)

                time_el = attributes.find(f"{ns}time")
                if time_el is not None:
                    beats = time_el.find(f"{ns}beats")
                    beat_type = time_el.find(f"{ns}beat-type")
                    if beats is not None and beat_type is not None:
                        score["timeSignature"] = [int(beats.text), int(beat_type.text)]

                key_el = attributes.find(f"{ns}key")
                if key_el is not None:
                    fifths_el = key_el.find(f"{ns}fifths")
                    if fifths_el is not None:
                        fifths = int(fifths_el.text)
                        score["keySignature"] = fifths_to_key.get(fifths, "C")

            # Check for tempo
            direction = measure.find(f"{ns}direction")
            if direction is not None:
                sound = direction.find(f".//{ns}sound")
                if sound is not None and sound.get("tempo"):
                    score["tempo"] = int(float(sound.get("tempo")))

            # Process notes
            current_beat = 1
            for note_el in measure.findall(f"{ns}note"):
                # Check if it's a rest
                is_rest = note_el.find(f"{ns}rest") is not None

                # Check if it's a chord (no forward movement)
                is_chord = note_el.find(f"{ns}chord") is not None

                # Get duration type
                type_el = note_el.find(f"{ns}type")
                dur_type = type_el.text if type_el is not None else "quarter"
                tone_dur = DURATION_MAP.get(dur_type, "4n")

                # Check for dot
                if note_el.find(f"{ns}dot") is not None:
                    tone_dur += "."

                # Get actual duration in divisions for beat tracking
                duration_el = note_el.find(f"{ns}duration")
                duration_divs = int(duration_el.text) if duration_el is not None else divisions

                if is_rest:
                    score["notes"].append({
                        "pitch": ["REST"],
                        "duration": tone_dur,
                        "measure": measure_num,
                        "beat": round(current_beat, 2),
                        "hand": hand,
                    })
                else:
                    # Get pitch
                    pitch_el = note_el.find(f"{ns}pitch")
                    if pitch_el is not None:
                        step = pitch_el.find(f"{ns}step").text  # C, D, E, etc.
                        octave = pitch_el.find(f"{ns}octave").text
                        alter_el = pitch_el.find(f"{ns}alter")
                        alter = int(float(alter_el.text)) if alter_el is not None else 0

                        accidental = ""
                        if alter == 1:
                            accidental = "#"
                        elif alter == -1:
                            accidental = "b"
                        elif alter == 2:
                            accidental = "##"
                        elif alter == -2:
                            accidental = "bb"

                        pitch_name = f"{step}{accidental}{octave}"

                        if is_chord and score["notes"]:
                            # Add to previous note's pitch array
                            score["notes"][-1]["pitch"].append(pitch_name)
                        else:
                            # Check for staccato
                            articulations = note_el.find(f".//{ns}articulations")
                            staccato = False
                            if articulations is not None:
                                staccato = articulations.find(f"{ns}staccato") is not None

                            note_data = {
                                "pitch": [pitch_name],
                                "duration": tone_dur,
                                "measure": measure_num,
                                "beat": round(current_beat, 2),
                                "hand": hand,
                            }
                            if staccato:
                                note_data["staccato"] = True

                            score["notes"].append(note_data)

                # Advance beat counter (skip for chord notes)
                if not is_chord:
                    beat_advance = duration_divs / divisions
                    current_beat += beat_advance

    return score


@app.post("/recognize")
async def recognize_sheet_music(request: ImageRequest):
    """Accept a base64 image, run homr OMR, return structured JSON."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))

        # Save to temp file (homr needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            # Run homr with proper config
            from homr.main import ProcessingConfig, process_image
            from homr.music_xml_generator import XmlGeneratorArguments

            config = ProcessingConfig(
                enable_debug=False,
                enable_cache=False,
                write_staff_positions=False,
                read_staff_positions=False,
                selected_staff=-1,
                use_gpu_inference=False,
            )
            xml_args = XmlGeneratorArguments()

            process_image(tmp_path, config, xml_args)

            # homr writes output next to the input file with .musicxml extension
            xml_path = Path(tmp_path).with_suffix(".musicxml")

            if not xml_path.exists():
                # Search for any .musicxml file in the temp dir
                import glob
                xml_files = sorted(
                    glob.glob(os.path.join(tempfile.gettempdir(), "*.musicxml")),
                    key=os.path.getmtime,
                    reverse=True,
                )
                if xml_files:
                    xml_path = Path(xml_files[0])

            if not xml_path.exists():
                raise FileNotFoundError("homr did not produce a MusicXML output")

            xml_string = xml_path.read_text(encoding="utf-8")

            # Convert MusicXML to our score format
            score = musicxml_to_score(xml_string)

            return score

        finally:
            # Cleanup temp files
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
