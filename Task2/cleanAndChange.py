import os
import json
import re
import pymorphy3

class CharacterReplacer:
    def __init__(self, terms_map_path):
        with open(terms_map_path, 'r', encoding='utf-8') as f:
            self.terms_map = json.load(f)

        self.morph = pymorphy3.MorphAnalyzer()
        self.section_pattern = re.compile(r'^###\s*(Галерея|Источники).*?(?=^###|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE)

        self.special_compounds = {
            "Ночной Дозор": "Войны стены",
            "Земли Ночного Дозора": "Земли Войнов Стены",
            "дом": "клан",
            "Дом": "Клан",
            "ДОМ": "КЛАН"
        }

        self._cached_replacement_dict = None

    def _remove_accents(self, text):
        accent_map = {
            'а́': 'а', 'е́': 'е', 'и́': 'и', 'о́': 'о', 'у́': 'у', 'ы́': 'ы', 'э́': 'э', 'ю́': 'ю', 'я́': 'я',
            'А́': 'А', 'Е́': 'Е', 'И́': 'И', 'О́': 'О', 'У́': 'У', 'Ы́': 'Ы', 'Э́': 'Э', 'Ю́': 'Ю', 'Я́': 'Я'
        }

        for accented, normal in accent_map.items():
            text = text.replace(accented, normal)

        return text

    def _analyze_word_forms(self, word, replacement):
        forms_dict = {}

        try:
            parsed_original = self.morph.parse(word)[0]
            parsed_replacement = self.morph.parse(replacement)[0]

            cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']

            for case in cases:
                try:
                    inflected_original = parsed_original.inflect({case})
                    if inflected_original:
                        original_form = inflected_original.word
                        inflected_replacement = parsed_replacement.inflect({case})
                        if inflected_replacement:
                            replacement_form = inflected_replacement.word
                            forms_dict[original_form.lower()] = replacement_form
                            forms_dict[original_form] = replacement_form.title()
                except:
                    continue

            try:
                plural_original = parsed_original.inflect({'plur'})
                if plural_original:
                    plural_replacement = parsed_replacement.inflect({'plur'})
                    if plural_replacement:
                        forms_dict[plural_original.word.lower()] = plural_replacement.word
                        forms_dict[plural_original.word] = plural_replacement.word.title()

                        for case in cases:
                            try:
                                original_plur_case = parsed_original.inflect({'plur', case})
                                if original_plur_case:
                                    replacement_plur_case = parsed_replacement.inflect({'plur', case})
                                    if replacement_plur_case:
                                        forms_dict[original_plur_case.word.lower()] = replacement_plur_case.word
                                        forms_dict[original_plur_case.word] = replacement_plur_case.word.title()
                            except:
                                continue
            except:
                pass

        except:
            pass

        forms_dict[word.lower()] = replacement.lower()
        forms_dict[word] = replacement
        forms_dict[word.title()] = replacement.title()

        if word.endswith(('б', 'в', 'г', 'д', 'ж', 'з', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ')):
            forms_dict[f"{word}ом"] = f"{replacement}ом"
            forms_dict[f"{word.title()}ом"] = f"{replacement.title()}ом"

        return forms_dict

    def _build_replacement_dict(self):
        if self._cached_replacement_dict:
            return self._cached_replacement_dict

        replacement_dict = {}

        for original, replacement in self.special_compounds.items():
            clean_original = self._remove_accents(original)
            replacement_dict[clean_original] = replacement
            replacement_dict[clean_original.lower()] = replacement.lower()
            replacement_dict[clean_original.title()] = replacement.title()

            if 'Ночной Дозор' in original:
                replacement_dict['Ночного Дозора'] = 'Войн стены'
                replacement_dict['ночного дозора'] = 'войн стены'
                replacement_dict['Ночному Дозору'] = 'Войнам стены'
                replacement_dict['Ночным Дозором'] = 'Войнами стены'
                replacement_dict['Ночном Дозоре'] = 'Войнах стены'
                replacement_dict['Ночные Дозоры'] = 'Войны стен'
                replacement_dict['Ночных Дозоров'] = 'Войн стен'
                replacement_dict['Ночным Дозорам'] = 'Войнам стен'
                replacement_dict['Ночными Дозорами'] = 'Войнами стен'
                replacement_dict['Ночных Дозорах'] = 'Войнах стен'

        for original, replacement in self.terms_map.items():
            clean_original = self._remove_accents(original)

            full_forms = self._analyze_word_forms(clean_original, replacement)
            for form, repl_form in full_forms.items():
                replacement_dict[form] = repl_form

            if ' ' in clean_original:
                parts = clean_original.split()
                repl_parts = replacement.split()

                if len(parts) == len(repl_parts):
                    for i, (orig_part, repl_part) in enumerate(zip(parts, repl_parts)):
                        part_forms = self._analyze_word_forms(orig_part, repl_part)
                        for form, repl_form in part_forms.items():
                            replacement_dict[form] = repl_form

                    all_part_forms = []
                    for orig_part, repl_part in zip(parts, repl_parts):
                        part_forms = self._analyze_word_forms(orig_part, repl_part)
                        all_part_forms.append(list(part_forms.items())[:3])

                    def generate_combos(index, current_orig, current_repl):
                        if index == len(parts):
                            replacement_dict[current_orig.strip()] = current_repl.strip()
                            return

                        for orig_form, repl_form in all_part_forms[index]:
                            new_orig = f"{current_orig} {orig_form}" if current_orig else orig_form
                            new_repl = f"{current_repl} {repl_form}" if current_repl else repl_form
                            generate_combos(index + 1, new_orig, new_repl)

                    generate_combos(0, "", "")

        additional_forms = {
            'Грейджоя': 'Дандрэгона',
            'Грейджою': 'Дандрэгону',
            'Грейджоем': 'Дандрэгоном',
            'Грейджое': 'Дандрэгоне',
            'Грейджои': 'Дандрэгоны',
            'Грейджоев': 'Дандрэгонов',
            'Грейджоям': 'Дандрэгонам',
            'Грейджоями': 'Дандрэгонами',
            'Грейджоях': 'Дандрэгонах',
            'Ланнистера': 'Натандема',
            'Ланнистеру': 'Натандему',
            'Ланнистером': 'Натандемом',
            'Ланнистер': 'Натандем',
            'Ланнистеры': 'Натандемы',
            'Ланнистеров': 'Натандемов',
            'Ланнистерам': 'Натандемам',
            'Ланнистерами': 'Натандемами',
            'Ланнистерах': 'Натандемах',
            'Таргариена': 'Лейскавена',
            'Таргариену': 'Лейскавену',
            'Таргариеном': 'Лейскавеном',
            'Таргариене': 'Лейскавене',
            'Таргариены': 'Лейскавены',
            'Таргариенов': 'Лейскавенов',
            'Таргариенам': 'Лейскавенам',
            'Таргариенами': 'Лейскавенами',
            'Таргариенах': 'Лейскавенах',
            'Старка': 'Стоункаттера',
            'Старку': 'Стоункаттеру',
            'Старком': 'Стоункаттером',
            'Старке': 'Стоункаттере',
            'Старки': 'Стоункаттеры',
            'Старков': 'Стоункаттеров',
            'Старкам': 'Стоункаттерам',
            'Старками': 'Стоункаттерами',
            'Старках': 'Стоункаттерах',
            'Роббом': 'Кристианом',
            'Теоном': 'Боривиком',
            'Джеймом': 'Кордуком',
            'Эддардом': 'Фаналиком',
            'Тирионом': 'Фирблом',
            'Бриенны': 'Улл',
            'Бриенне': 'Улл',
            'Бриенной': 'Улл',
            'Бриенну': 'Улл',
            'Арью': 'Дентрату',
            'Арьей': 'Дентратой',
        }

        for form, replacement in additional_forms.items():
            replacement_dict[form] = replacement
            replacement_dict[form.lower()] = replacement.lower()
            replacement_dict[form.title()] = replacement.title()

        self._cached_replacement_dict = replacement_dict
        return replacement_dict

    def _create_advanced_pattern(self, replacement_dict):
        patterns = list(replacement_dict.keys())
        patterns.sort(key=len, reverse=True)
        escaped_patterns = [re.escape(p) for p in patterns]
        pattern = r'\b(' + '|'.join(escaped_patterns) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def _replace_text(self, text, replacement_dict, pattern):
        def replace_match(match):
            matched_text = match.group(0)

            if matched_text in replacement_dict:
                return replacement_dict[matched_text]
            elif matched_text.lower() in replacement_dict:
                replacement = replacement_dict[matched_text.lower()]
                if matched_text.istitle():
                    return replacement.title()
                elif matched_text.isupper():
                    return replacement.upper()
                else:
                    return replacement
            elif matched_text.title() in replacement_dict:
                return replacement_dict[matched_text.title()]

            return matched_text

        result = pattern.sub(replace_match, text)
        result = re.sub(r'Земли\s+Ночного\s+Дозора', 'Земли Войнов Стены', result, flags=re.IGNORECASE)
        result = re.sub(r'великих\s+домов', 'великих кланов', result, flags=re.IGNORECASE)
        result = re.sub(r'Великих\s+Домов', 'Великих Кланов', result, flags=re.IGNORECASE)

        return result

    def _remove_sections(self, text):
        return self.section_pattern.sub('', text)

    def process_text(self, text):
        text = self._remove_accents(text)
        replacement_dict = self._build_replacement_dict()
        pattern = self._create_advanced_pattern(replacement_dict)
        text = self._replace_text(text, replacement_dict, pattern)
        text = self._remove_sections(text)
        return text

    def process_filename(self, filename):
        name, ext = os.path.splitext(filename)
        parts = re.split(r'[_\-\s]+', name)
        processed_parts = []
        for part in parts:
            processed_part = self.process_text(part)
            processed_parts.append(processed_part)
        processed_name = '_'.join(processed_parts)
        return processed_name + ext

    def process_file(self, input_path, output_dir):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            processed_content = self.process_text(content)
            original_filename = os.path.basename(input_path)
            processed_filename = self.process_filename(original_filename)
            output_path = os.path.join(output_dir, processed_filename)

            counter = 1
            base_name, ext = os.path.splitext(processed_filename)
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                counter += 1

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)

            return True

        except Exception as e:
            return False

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

        input_files = []
        for file in os.listdir(input_dir):
            if file.endswith('.txt'):
                input_files.append(os.path.join(input_dir, file))

        if not input_files:
            return

        for input_file in input_files:
            self.process_file(input_file, output_dir)


def main():
    terms_map_path = "terms_map.json"
    input_directory = "characters_clean"
    output_directory = "knowledge_base"

    if not os.path.exists(input_directory) or not os.path.exists(terms_map_path):
        return

    replacer = CharacterReplacer(terms_map_path)
    replacer.process_directory(input_directory, output_directory)


if __name__ == "__main__":
    main()