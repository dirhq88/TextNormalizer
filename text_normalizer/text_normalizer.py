from typing import Dict, List, Tuple
import hangul_dtw
import re
from jamo import h2j, j2h

class TextNormalizer:
    #TYPE_NO_CHANGE = 'no_change'
    TYPE_CHANGE = 'change'
    TYPE_MERGE = 'merge'
    TYPE_CHANGE_WITH_PITCH = 'change_with_pitch_change'
    TYPE_MERGE_NO_PITCH = 'merge_no_pitch_change' # without pitch change
    TYPE_SILENCE = 'silence'  # for spaces
    
    def __init__(self):
        """텍스트 정규화를 위한 클래스를 초기화합니다."""
        self.dtw = hangul_dtw.hangul_DTW
        self.vowel_set_map = {
            'ᅪ': 'ᅡ', 'ᅡ': 'ᅡ', 'ᅣ': 'ᅡ', 'ᅱ': 'ᅵ', 'ᅵ': 'ᅵ',
            'ᅯ': 'ᅥ', 'ᅥ': 'ᅥ', 'ᅧ': 'ᅥ', 'ᅭ': 'ᅩ', 'ᅩ': 'ᅩ',
            'ᅲ': 'ᅮ', 'ᅮ': 'ᅮ', 'ᅰ': 'ᅦ', 'ᅫ': 'ᅦ', 'ᅬ': 'ᅦ',
            'ᅨ': 'ᅦ', 'ᅦ': 'ᅦ', 'ᅤ': 'ᅦ', 'ᅢ': 'ᅢ',
            'ᅴ': 'ᅵ', 'ᅳ': 'ᅳ'
        }

    def get_space_positions(self, text: str) -> List[int]:
        """텍스트에서 공백의 위치를 찾아 반환합니다."""
        return [i for i, char in enumerate(text) if char.isspace()]

    def find_space_insertion_points(self, raw_text: str, normalization_infos: List[Dict]) -> List[Dict]:
        """정규화된 텍스트에 공백을 삽입할 위치를 찾습니다."""
        space_positions_in_original_raw = self.get_space_positions(raw_text)
        insertion_points = []

        # 맵1: 원본 raw_text의 (공백 아닌) 문자 인덱스 -> raw_text_no_space의 인덱스
        map_original_raw_idx_to_no_space_idx = {}
        current_no_space_idx = 0
        for i, char_original in enumerate(raw_text):
            if not char_original.isspace():
                map_original_raw_idx_to_no_space_idx[i] = current_no_space_idx
                current_no_space_idx += 1
        
        # 맵2: raw_text_no_space의 문자 인덱스 -> 그 문자가 기여한 마지막 normalized_idx
        map_no_space_to_last_normalized_idx = {}
        for info in normalization_infos:
            if info.get('type') == self.TYPE_SILENCE: # Should not exist yet, but defensive
                continue
            current_normalized_idx_for_this_info = info['normalized_idx']
            if 'raw_indices' in info: # 1:N (여러 raw_no_space 문자가 이 info의 norm_syllable 하나를 만듬)
                for raw_idx_ns in info['raw_indices']:
                    map_no_space_to_last_normalized_idx[raw_idx_ns] = current_normalized_idx_for_this_info
            elif 'raw_idx' in info: # 1:1
                raw_idx_ns = info['raw_idx']
                map_no_space_to_last_normalized_idx[raw_idx_ns] = current_normalized_idx_for_this_info

        for space_pos_in_original in space_positions_in_original_raw:
            last_preceding_non_space_original_idx = -1
            for k_rev in range(space_pos_in_original - 1, -1, -1):
                if not raw_text[k_rev].isspace():
                    last_preceding_non_space_original_idx = k_rev
                    break
            
            current_pos_to_insert = 0
            if last_preceding_non_space_original_idx != -1:
                no_space_idx = map_original_raw_idx_to_no_space_idx.get(last_preceding_non_space_original_idx)
                if no_space_idx is not None:
                    norm_idx = map_no_space_to_last_normalized_idx.get(no_space_idx)
                    if norm_idx is not None:
                        current_pos_to_insert = norm_idx + 1
                    # else: Error / no norm_idx for this no_space_idx (should not happen if maps complete)
                # else: Error / no_space_idx for this original_raw_idx (should not happen)
            else:
                # All preceding chars are spaces, or this is the first char (and it's a space)
                count_preceding_spaces = 0
                for k_count in range(space_pos_in_original):
                    if raw_text[k_count].isspace():
                        count_preceding_spaces += 1
                current_pos_to_insert = count_preceding_spaces

            insertion_points.append({
                'position': current_pos_to_insert,
                'raw_idx': space_pos_in_original 
            })
        
        return sorted(insertion_points, key=lambda x: (x['position'], x['raw_idx']))

    def insert_spaces(self, normalized_texts: List[str], normalized_pitches: List[int], 
                     normalization_infos: List[Dict], insertion_points: List[Dict]) -> Tuple[List[str], List[int], List[Dict]]:
        """정규화된 텍스트에 공백을 삽입합니다."""
        # 삽입 위치를 역순으로 정렬하여 인덱스 변화를 방지
        for point in reversed(insertion_points):
            pos = point['position']
            normalized_texts.insert(pos, ' ')
            normalized_pitches.insert(pos, 0)
            
            # 공백에 대한 normalization_info 추가
            space_info = {
                'type': self.TYPE_SILENCE,
                'normalized_idx': pos,
            }
            normalization_infos.insert(pos, space_info)
            
            # 이후의 normalized_idx 업데이트
            for info in normalization_infos[pos + 1:]:
                info['normalized_idx'] += 1
        
        return normalized_texts, normalized_pitches, normalization_infos

    def normalize_text(self, gt_text: str, raw_text: str, pitch_sequence: List[int], insert_spaces: bool = True) -> Dict:
        """
        입력 텍스트를 정규화하여 반환합니다.
        
        Args:
            gt_text (str): 정답 텍스트
            raw_text (str): 원본 텍스트
            pitch_sequence (List[int]): raw_text의 각 음절에 대응하는 피치 값 리스트
            
        Returns:
            Dict: {
                'normalized_text': List[str],  # 정규화된 텍스트
                'normalized_pitch': List[int],  # 정규화된 피치
                'normalization_info': List[Dict]  # 정규화 정보
            }
        """
        # 1. 전처리 (공백 제거)
        gt_text = re.sub(r'[^ㄱ-ㅎ가-힣]+', '', gt_text)
        raw_text_no_space = re.sub(r'[^ㄱ-ㅎ가-힣]+', '', raw_text)
        
        gt_syllables = list(gt_text)
        raw_syllables = list(raw_text_no_space)
        
        # 2. DTW를 통한 정렬
        _, _, _, syllable_mapping = self.dtw(gt_text, raw_text)
        
        # 3. 정규화 결과를 저장할 리스트들
        normalized_texts = []
        normalized_pitches = []
        normalization_infos = []
        current_idx = 0  # 현재 정규화된 텍스트의 인덱스
        
        # 4. syllable_mapping을 순회하며 정규화 수행
        for gt_idx, raw_indices in syllable_mapping.items():
            if len(raw_indices) == 1:
                # 1:1 매핑 처리
                texts, pitches, infos = self.normalize_one_to_one_mapping(gt_syllables, gt_idx, raw_syllables, raw_indices, pitch_sequence, current_idx)
                
            else:
                # 1:N 매핑 처리
                texts, pitches, infos = self.normalize_one_to_many_mapping(gt_syllables, gt_idx, raw_syllables, raw_indices, pitch_sequence, current_idx)
                
            normalized_texts.extend(texts)
            normalized_pitches.extend(pitches)
            normalization_infos.extend(infos)
            current_idx += len(texts)  # 인덱스 업데이트
        
        if insert_spaces:
            # 5. 공백 복원
            insertion_points = self.find_space_insertion_points(raw_text, normalization_infos)
            normalized_texts, normalized_pitches, normalization_infos = self.insert_spaces(
                normalized_texts, normalized_pitches, normalization_infos, insertion_points
            )
        
        return {
            'normalized_texts': normalized_texts,
            'normalized_pitches': normalized_pitches,
            'normalization_infos': normalization_infos
        }

    def is_pitch_change(self, pitch_sequence: List[int], raw_indices: List[int]) -> bool:
        """주어진 구간에 피치 변화가 있는지 확인합니다."""
            
        first_pitch = pitch_sequence[raw_indices[0]]
        for i in range(raw_indices[0] + 1, raw_indices[-1] + 1):
            if pitch_sequence[i] != first_pitch:
                return True
        return False

    def normalize_one_to_one_mapping(self, gt_syllables: List[str], gt_idx: int, raw_syllables: List[str], 
                                  raw_indices: List[int], pitch_sequence: List[int], current_idx: int) -> Tuple[List[str], List[int], List[Dict]]:
        """1:1 매핑을 처리합니다."""
        gt_syllable = gt_syllables[gt_idx]
        
        normalized_info = {
            'type': self.TYPE_CHANGE,   
            'normalized_syllable': gt_syllable,
            'gt_idx': gt_idx,
            'raw_idx': raw_indices[0],
            'normalized_idx': current_idx
        }
        return [gt_syllable], [pitch_sequence[raw_indices[0]]], [normalized_info]

    def normalize_one_to_many_mapping(self, gt_syllables: List[str], gt_idx: int, raw_syllables: List[str], 
                                   raw_indices: List[int], pitch_sequence: List[int], current_idx: int) -> Tuple[List[str], List[int], List[Dict]]:
        """1:N 매핑을 처리합니다."""
        if not self.is_pitch_change(pitch_sequence, raw_indices):
            gt_syllable = gt_syllables[gt_idx]
            
            normalized_info = {
                'type': self.TYPE_MERGE,
                'normalized_syllable': gt_syllable,
                'gt_idx': gt_idx,
                'raw_indices': raw_indices,
                'normalized_idx': current_idx
            }
            return [gt_syllable], [pitch_sequence[raw_indices[0]]], [normalized_info]   
        else:
            split_indices = self.split_indices_by_pitch_change(raw_indices, pitch_sequence)
            return self.normalize_one_to_many_mapping_by_pitch_change(gt_syllables, gt_idx, raw_syllables, split_indices, pitch_sequence, current_idx)

    def split_indices_by_pitch_change(self, raw_indices: List[int], pitch_sequence: List[int]) -> List[Tuple[int, int]]:
        '''
        Example:
            raw_indices: [0, 1, 2, 3]
            pitch_sequence: [60, 60, 61, 61]
            return: [(0, 1), (2, 3)]
        '''
        split_points = []
        for i in range(raw_indices[0], raw_indices[-1]):
            if pitch_sequence[i] != pitch_sequence[i+1]:
                split_points.append(i)

        split_indices = []
        start = raw_indices[0]
        for point in split_points:
            split_indices.append((start, point))
            start = point + 1
        split_indices.append((start, raw_indices[-1]))
        
        return split_indices

    def normalize_one_to_many_mapping_by_pitch_change(self, gt_syllables: List[str], gt_idx: int, raw_syllables: List[str], 
                                                    split_indices: List[Tuple[int, int]], pitch_sequence: List[int], current_idx: int) -> Tuple[List[str], List[int], List[Dict]]:
        # 각 청크 처리
        normalized_texts = []
        normalized_pitches = []
        normalization_infos = []
        
        gt_jamos = h2j(gt_syllables[gt_idx])  # GT 음절을 자모로 분리
        
        for i, (start, end) in enumerate(split_indices):
            
            if i == 0:  # 첫 번째 청크
                normalized_text = j2h(gt_jamos[0], gt_jamos[1])  # 초성 + 중성
            elif i == len(split_indices) - 1:  # 마지막 청크
                if len(gt_jamos) > 2:  # 종성이 있는 경우
                    normalized_text = j2h('ㅇ', self.vowel_set_map[gt_jamos[1]], gt_jamos[2])
                else:  # 종성이 없는 경우
                    normalized_text = j2h('ㅇ', self.vowel_set_map[gt_jamos[1]])
            else:  # 중간 청크
                normalized_text = j2h('ㅇ', self.vowel_set_map[gt_jamos[1]])
            
            normalized_texts.append(normalized_text)
            normalized_pitches.append(pitch_sequence[start])
            normalization_infos.append({
                'type': self.TYPE_MERGE if len(list(range(start, end + 1))) > 1 else self.TYPE_CHANGE,
                'normalized_syllable': normalized_text,
                'gt_idx': gt_idx,
                'raw_indices': list(range(start, end + 1)),
                'normalized_idx': current_idx + i
            })
        
        return normalized_texts, normalized_pitches, normalization_infos