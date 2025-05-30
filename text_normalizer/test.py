import pandas as pd
from text_normalizer import TextNormalizer
import json

split_json_dir = "./split_json"
normalized_json_dir = "./normalized_json"

file_name = "SINGER_46_10TO29_NORMAL_FEMALE_BALLAD_C1925"
number = "04"

split_json_filepath = f"{split_json_dir}/{file_name}_{number}.json"
with open(split_json_filepath, "r") as j:
        split_json_meta = json.load(j)
split_json_notes = split_json_meta['notes']
Split_json = pd.json_normalize(split_json_meta, record_path=['notes'])

normalized_json_filepath = f"{normalized_json_dir}/{file_name}_{number}.json"
with open(normalized_json_filepath, "r") as j:
        normalized_json_meta = json.load(j)
normalized_json_notes = normalized_json_meta['notes']
Normalized_json = pd.json_normalize(normalized_json_meta, record_path=['notes'])

whisper_dir = "./whisper_results"
whisper_filepath = f"{whisper_dir}/{file_name}_{number}.txt"
with open(whisper_filepath, "r") as j:
    whisper_text = j.read()

split_json_text = ''
split_pitch_sequence = []
normalized_json_text = ''
normalized_pitch_sequence = []

for note in split_json_notes:
    split_json_text += note['lyric']
    split_pitch_sequence.append(note['midi_num'])

for note in normalized_json_notes:
    normalized_json_text += note['lyric']
    normalized_pitch_sequence.append(note['midi_num'])

split_pitch_sequence = [x for x in split_pitch_sequence if x != 0]
normalized_pitch_sequence = [x for x in normalized_pitch_sequence if x != 0]
split_json_text = split_json_text.lstrip().rstrip()

print(f"gt text: {whisper_text}")
print(f"raw text: {split_json_text}")
print(f"raw pitch: {split_pitch_sequence}")
print('--------------------------------')
#print(normalized_json_text)
#print(normalized_pitch_sequence)

text_normalizer = TextNormalizer()

results = text_normalizer.normalize_text(whisper_text, split_json_text, split_pitch_sequence, True)
print('--------------------------------')
print(results['normalized_texts'])
print(results['normalized_pitches'])
#print(results['normalization_infos'])

for info in results['normalization_infos']:
    print(info)
    
