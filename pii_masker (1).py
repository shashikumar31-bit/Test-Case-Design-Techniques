import json
import os
import re
import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
import random

from pydub import AudioSegment
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

def generate_beep_wave(beep_path: str, frequency: float = 1000.0, duration: float = 1.0, sample_rate: int = 44100, amplitude: float = 0.8) -> None:
    """Generate a sine wave beep sound and save as WAV file."""
    try:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
        wavfile.write(beep_path, sample_rate, audio)
    except Exception as e:
        raise RuntimeError(f"Failed to generate beep.wav: {e}")

def detect_language(transcription: List[Dict], surname_file: str = "filipino_surnames.txt") -> str:
    """Detect language based on percentage of Filipino words in transcription."""
    # Load Filipino surnames
    surnames = set()
    try:
        if os.path.exists(surname_file):
            with open(surname_file, "r", encoding="utf-8") as f:
                surnames = {line.strip().lower() for line in f if line.strip()}
    except Exception as e:
        print(f"Error loading surnames: {e}")

    # Common Filipino words (partial list for detection)
    filipino_words = {
        "ako", "ikaw", "siya", "kami", "kayo", "sila", "ito", "iyan", "iyon",
        "sa", "ng", "at", "para", "mga", "nang", "din", "rin", "ang", "na",
        "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam",
        "po", "opo", "hindi", "oo", "salamat", "magandang", "araw", "gabi",
        "pa", "ba", "kung", "dahil", "pero", "subalit", "kasi", "eh"
    } | surnames  # Include surnames as Filipino words

    total_words = 0
    filipino_word_count = 0

    for entry in transcription:
        text = entry.get("dialogue", "")
        words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
        total_words += len(words)
        filipino_word_count += sum(1 for word in words if word in filipino_words)

    # Calculate percentage
    filipino_percentage = (filipino_word_count / total_words * 100) if total_words > 0 else 0
    print(f"Filipino words: {filipino_word_count}/{total_words} ({filipino_percentage:.2f}%)")

    # Threshold: ≥30% Filipino words → Filipino call
    return ("filipino" if filipino_percentage >= 3 else "english", filipino_percentage)


import re
import random
from typing import List, Dict, Tuple
import os

class FilipinoPIIMasker:
    """
    PII masker for Filipino/Tagalog call transcriptions.
    
    GDPR-aligned: Only masks data that can be used to trace back to the customer.
    Does NOT mask: bank/company names, agent names, generic numbers, standalone months.
    """

    # Allowlist: words that should NEVER be masked even if they match the surname list.
    # Includes common English/Filipino words, bank names, company names, greetings, etc.
    ALLOWLIST = {
        # Common English words that appear in name lists
        "may", "mark", "rose", "angel", "grace", "joy", "hope", "faith", "charity",
        "sun", "star", "sky", "crystal", "diamond", "ruby", "pearl", "amber",
        "summer", "winter", "spring", "autumn", "dawn", "eve", "april", "june",
        "august", "miles", "chase", "hunter", "mason", "prince", "king", "queen",
        "royal", "noble", "sterling", "chance", "stone", "cash", "bill", "frank",
        "will", "art", "ray", "dale", "glen", "dean", "grant", "wade", "lance",
        "heath", "cliff", "reed", "ford", "banks", "lane", "lake", "brook",
        "candy", "cherry", "ginger", "holly", "ivy", "lily", "poppy", "daisy",
        "violet", "iris", "jasmine", "heather", "fern", "olive", "sage", "hazel",
        "love", "mercy", "honor", "glory", "destiny", "harmony", "melody",
        "liberty", "justice", "haven", "trinity", "genesis", "phoenix",
        # Common Filipino words that could be names
        "po", "opo", "na", "ba", "pa", "ang", "mga", "sa", "ng", "at",
        "para", "din", "rin", "lang", "naman", "siya", "ako", "ikaw",
        "kami", "kayo", "sila", "ito", "iyan", "iyon", "dito", "diyan",
        "lima", "isa", "dalawa", "tatlo", "apat", "anim", "pito", "walo", "siyam",
        # Bank and financial institution names
        "bdo", "bpi", "metrobank", "landbank", "pnb", "rcbc", "unionbank",
        "chinabank", "eastwest", "securitybank", "maybank", "citi", "citibank",
        "hsbc", "standard chartered", "deutsche", "barclays", "wells fargo",
        "hdfc", "icici", "sbi", "axis", "kotak", "idfc", "bandhan",
        "mahindra", "bajaj", "tata", "reliance", "adani",
        # Company/org names commonly appearing in calls
        "globe", "smart", "pldt", "meralco", "jollibee", "cebu pacific",
        "gcash", "paymaya", "lazada", "shopee",
        # Generic call center terms
        "sir", "ma'am", "madam", "maam", "miss", "mister",
        "customer", "agent", "representative", "associate", "supervisor",
        "manager", "team", "department", "service", "support",
        "account", "payment", "balance", "loan", "emi", "finance",
        "bank", "credit", "debit", "card", "number", "amount",
        "collection", "recovery", "settlement", "overdue", "due",
        "good", "morning", "afternoon", "evening", "night",
        "thank", "thanks", "okay", "yes", "no", "please", "sorry",
        "hello", "hi", "bye", "welcome",
        # Operational terms
        "option", "press", "hold", "transfer", "connect", "call",
        "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
        "first", "second", "third", "last", "next", "previous",
    }

    def __init__(
        self,
        surname_file: str = "allnamesnew.txt",
        regions_file: str = "philippines_regions.txt"
    ):
        # storage for downstream masking results
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []

        # load raw lists
        self.surnames = self.load_surnames(surname_file)    # returns set
        self.regions = self.load_regions(regions_file)

        # Filter surnames: remove any that are in the allowlist
        self.surnames = self.surnames - self.ALLOWLIST

        # compile one big surname regex (case‑insensitive, word boundaries)
        escaped = [re.escape(s) for s in self.surnames if s.strip()]
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"
        self.surname_re = re.compile(pattern, re.IGNORECASE)

        # simple list of common first names (for fictitious name substitution)
        self.first_names = [
            "Mark", "Anna", "Jose", "Maria", "John", "Clara", "Pedro", "Liza",
            "Ramon", "Teresa", "Ben", "Sofia", "Luis", "Emma", "Carlos"
        ]

        # build GDPR-relevant PII patterns (context-aware)
        physical_address_pattern = (
            r"\b(?:" + "|".join(re.escape(rgn) for rgn in self.regions) + r")\b"
        )
        self.pii_patterns = [
            # --- Phone numbers: 10-11+ digit sequences, optionally with separators ---
            {
                "entity": "PHONE_NUMBER",
                "pattern": (
                    # Standard digit phone numbers with optional country code
                    r"((?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4})"
                    r"|"
                    # Phone numbers spoken as words (context required)
                    r"(?:(?:number|numero|number is|numero is|contact|phone|mobile|cell)\s)"
                    r"((?:(?:zero|one|two|three|four|five|six|seven|eight|nine)\s*){7,})"
                ),
                "context": None
            },
            # --- Account / Reference numbers: digit sequences in account context ---
            {
                "entity": "ACCOUNT_NUMBER",
                "pattern": (
                    # CF-style reference numbers
                    r"(\bCF[- ]?\d{4,}(?:\s*dash\s*\d+)?\b)"
                    r"|"
                    # Number sequences near account-related keywords
                    r"(?:(?:account|reference|loan|contract|ref)\s(?:number\s)?(?:is\s)?)(\d{4,}(?:[- ]\d+)*)"
                    r"|"
                    r"(\d{4,}(?:[- ]\d+)*)(?=\s*(?:is your|is the|account|reference|loan|contract))"
                ),
                "context": r"\b(?:account|reference|loan|contract|ref|CF)\b"
            },
            # --- Credit card numbers ---
            {
                "entity": "CREDIT_CARD",
                "pattern": (
                    # Full credit card numbers (13-19 digits with optional separators)
                    r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7}\b"
                    r"|"
                    # Partial card numbers in context
                    r"(?:(?:credit|debit)\s*card\s*(?:ending|last\s*\d\s*digits?)\s*(?:is\s*|with\s*|in\s*)?)\d{3,4}"
                    r"|"
                    r"(?:ending\s*(?:in|with)\s*)\d{3,4}(?=\s|$|[.,])"
                ),
                "context": None
            },
            # --- Email addresses ---
            {
                "entity": "EMAIL_ADDRESS",
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "context": None
            },
            # --- Partial Numbers (last 4 digits) ---
            {
                "entity": "PARTIAL_NUMBER",
                "pattern": (
                    r"(?:last\s*\d+\s*digits?|ending\s*(?:in|with))\s*"
                    r"(?:registered\s*with\s*[a-zA-Z]+\s*)?"
                    r"(?:is\s*|are\s*|with\s*|in\s*)?(\d{3,4})"
                ),
                "context": None
            },
            # --- Physical addresses (region/city names from the regions file) ---
            {
                "entity": "PHYSICAL_ADDRESS",
                "pattern": physical_address_pattern,
                "context": None
            },
            # --- Date of Birth: full date expressions only (day + month + year) ---
            {
                "entity": "DATE_OF_BIRTH",
                "pattern": (
                    # Numeric date formats: DD/MM/YYYY, MM-DD-YYYY, etc.
                    r"(\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b)"
                    r"|"
                    # Month DD, YYYY or DD Month YYYY
                    r"(\b(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December|"
                    r"Enero|Pebrero|Marso|Abril|Mayo|Hunyo|Hulyo|Agosto|"
                    r"Setyembre|Oktubre|Nobyembre|Disyembre)"
                    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}\b)"
                    r"|"
                    r"(\b\d{1,2}(?:st|nd|rd|th)?\s+"
                    r"(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December|"
                    r"Enero|Pebrero|Marso|Abril|Mayo|Hunyo|Hulyo|Agosto|"
                    r"Setyembre|Oktubre|Nobyembre|Disyembre)"
                    r",?\s*\d{2,4}\b)"
                    r"|"
                    # Month + Year (only in DOB context)
                    r"(?:(?:born|birthday|birth date|date of birth|birthdate|"
                    r"kapanganakan|petsa ng kapanganakan)\s(?:is\s)?(?:on\s)?)"
                    r"((?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+\d{2,4})"
                ),
                "context": None
            },
            # --- National ID / IC numbers ---
            {
                "entity": "NATIONAL_ID",
                "pattern": (
                    # SSS format: XX-XXXXXXX-X
                    r"(\b\d{2}-\d{7}-\d\b)"
                    r"|"
                    # TIN format: XXX-XXX-XXX-XXX
                    r"(\b\d{3}-\d{3}-\d{3}-\d{3}\b)"
                    r"|"
                    # Passport: Letter followed by 7-8 digits
                    r"(\b[A-Z]\d{7,8}\b)"
                    r"|"
                    # Generic ID in context (requires word boundaries and at least one digit)
                    r"(?:\b(?:IC|NRIC|SSS|TIN|passport|license|id)\b\s+(?:number\s+)?(?:is\s+)?)(?=.*?\d)([A-Za-z0-9\-]{6,})"
                ),
                "context": None
            },
            # --- Parent / family member details (security question answers) ---
            {
                "entity": "PARENT_DETAILS",
                "pattern": (
                    r"(?:(?:mother'?s?|father'?s?|nanay|tatay|ina|ama|"
                    r"maiden|parents?'?)\s*(?:name|maiden name|last name|surname)"
                    r"\s*(?:is|ay|po)?\s*)"
                    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
                ),
                "context": r"\b(?:mother|father|nanay|tatay|ina|ama|maiden|parent)\b"
            },
            # --- Smart Month Masking ---
            {
                "entity": "MONTH",
                "pattern": (
                    r"(\b(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December|"
                    r"Enero|Pebrero|Marso|Abril|Mayo|Hunyo|Hulyo|Agosto|"
                    r"Setyembre|Oktubre|Nobyembre|Disyembre)\b)"
                ),
                "context": None
            },
        ]

        # Patterns to capture customer names via context phrases.
        # These only match names spoken BY the agent ABOUT the customer,
        # or names the customer gives as identity verification.
        self.name_patterns = [
            # Customer introducing themselves: "my name is Santos"
            r"(?i)\b(?:my name is|I'm|I am)\s+(?-i:([A-Z][a-z]+))\b",
            # Agent asking: "what is your good name? Santos"
            r"(?i)\b(?:what is your good name)\s*[,?\s]*(?-i:([A-Z][a-z]+))\b",
            # Agent addressing customer with title: "Mr. Santos", "Mrs. Cruz"
            r"(?i)\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss)\s+(?-i:([A-Z][a-z]+))\b",
            # Agent confirming identity: "speaking with Santos"
            r"(?i)\b(?:speaking with|talking to|verify|confirm)\s+(?:Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+|Miss\s+)?(?-i:([A-Z][a-z]+))\b",
            # "Am I speaking with..." pattern
            r"(?i)\bam I speaking (?:with|to)\s+(?:Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+|Miss\s+)?(?-i:([A-Z][a-z]+))\b",
        ]

        # Context patterns that indicate a surname is being spoken as a person's name
        # (not just a common word). A surname match must be near one of these contexts.
        self.name_context_patterns = [
            r"(?i)\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss)\s*$",       # Title immediately before
            r"(?i)\b(?:name|pangalan)\b",                     # Word "name" nearby
            r"(?i)\b(?:speaking with|talking to)\b",          # Agent referring to customer
            r"(?i)\b(?:verify|confirm|verification)\b",       # Verification context
            r"(?i)\b(?:account\s*(?:holder|name|of))\b",      # Account holder
            r"(?i)\b(?:mother|father|maiden|nanay|tatay)\b",  # Family context
        ]

    def load_surnames(self, surname_file: str) -> set:
        surnames = set()
        try:
            if os.path.exists(surname_file):
                with open(surname_file, "r", encoding="utf-8") as f:
                    surnames = {line.strip().lower() for line in f if line.strip()}
                print(f"Loaded {len(surnames)} surnames from {surname_file}")
            else:
                print(f"Warning: Surname file {surname_file} not found.")
        except Exception as e:
            print(f"Error loading surnames: {e}")
        return surnames

    def load_regions(self, regions_file: str) -> set:
        regions = set()
        try:
            if os.path.exists(regions_file):
                with open(regions_file, "r", encoding="utf-8") as f:
                    regions = {line.strip() for line in f if line.strip()}
                print(f"Loaded {len(regions)} regions from {regions_file}")
            else:
                print(f"Warning: Regions file {regions_file} not found.")
        except Exception as e:
            print(f"Error loading regions: {e}")
        return regions

    def _is_in_name_context(self, text: str, match_start: int, match_end: int, window: int = 80) -> bool:
        """Check if a surname match appears in a name-like context within a text window."""
        # Get surrounding text window
        context_start = max(0, match_start - window)
        context_end = min(len(text), match_end + window)
        context_text = text[context_start:context_end]

        for ctx_pattern in self.name_context_patterns:
            if re.search(ctx_pattern, context_text):
                return True

        # Also check if immediately preceded by a capitalized word (likely a first name)
        before_text = text[max(0, match_start - 20):match_start].rstrip()
        if re.search(r'[A-Z][a-z]+$', before_text):
            return True

        return False

    def _add_pii_result(self, pii_results, occupied, mappings, start, end, original, entity, text, start_time, end_time):
        """Helper to record a PII detection result."""
        placeholder = f"[{entity}]"
        mappings.setdefault(placeholder, []).append(original)

        length = len(text)
        dur = end_time - start_time
        if length > 0:
            t0 = start_time + (start / length) * dur
            t1 = start_time + (end / length) * dur
        else:
            t0, t1 = start_time, end_time
        self.pii_timelines.append((t0, t1, original, entity))

        pii_results.append((start, end, original, entity))
        for i in range(start, end):
            if i < len(occupied):
                occupied[i] = True

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []

        for entry in transcription:
            text = entry["dialogue"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            mappings: Dict[str, List[str]] = {}
            modified_text = text
            pii_results: List[Tuple[int,int,str,str]] = []

            # Boolean mask for occupied character positions
            occupied = [False] * len(modified_text)

            # --- Step 3: GDPR-relevant PII patterns ---
            for pii_type in self.pii_patterns:
                entity = pii_type["entity"]
                pattern = pii_type["pattern"]
                ctx = pii_type.get("context")
                if ctx and not re.search(ctx, modified_text, re.IGNORECASE):
                    continue
                for match in re.finditer(pattern, modified_text, re.IGNORECASE):
                    if match.lastindex and match.lastindex >= 1:
                        s, e = match.start(match.lastindex), match.end(match.lastindex)
                        original = match.group(match.lastindex)
                    else:
                        s, e = match.start(), match.end()
                        original = match.group(0)
                        
                    if any(occupied[i] for i in range(s, e)):
                        continue

                    # Skip non-PII words (greetings, time-of-day, etc.)
                    if original.lower() in self.ALLOWLIST:
                        continue
                        
                    # Smart logic for MONTH to filter out false positives
                    if entity == "MONTH":
                        before = modified_text[max(0, s-30):s].lower()
                        # Only mask if near DOB context
                        if not re.search(r'\b(born|birth|dob|birthday|kapanganakan)\b', before):
                            continue
                            
                    # Filter out PHYSICAL_ADDRESS false positives
                    if entity == "PHYSICAL_ADDRESS":
                        word_lower = original.lower()
                        before = modified_text[max(0, s-30):s].lower()
                        
                        # 1. Person name check (titles)
                        if re.search(r'\b(mr|mrs|ms|miss|sir|maam|madam)\.?\s+(?:[a-z]+\s+){0,2}$', before):
                            continue
                            
                        # 2. Require context for ALL single-word physical addresses
                        # This prevents common words (Care, Making, Upon) and surnames (Maya, Madrid) from false matching
                        if len(word_lower.split()) == 1:
                            location_context = r'\b(live|residing|address|city|municipality|brgy|barangay|in|at|from|to|street|st|ave|avenue|province|region)(?:\s+(?:is|of))?\s+$'
                            if not re.search(location_context, before):
                                continue
                                
                        # 3. Collection Agency check (e.g. SP Madrid)
                        if word_lower == "madrid" and re.search(r'\b(sp|s\.p\.?)\s+$', before):
                            continue
                                
                    # Skip very short matches for non-specific patterns
                    if entity in ("PHONE_NUMBER",) and len(original.replace(" ", "").replace("-", "")) < 7:
                        continue
                    self._add_pii_result(pii_results, occupied, mappings, s, e,
                                         original, entity, modified_text, start_time, end_time)

            # --- Apply all replacements in reverse order ---
            for s, e, orig, ent in sorted(pii_results, key=lambda x: x[0], reverse=True):
                placeholder = f"[{ent}]"
                modified_text = modified_text[:s] + placeholder + modified_text[e:]

            # finalize
            masked_entry = entry.copy()
            masked_entry["dialogue"] = modified_text
            self.masked_transcription.append(masked_entry)
            self.pii_mappings.append((modified_text, mappings))

        self._propagate_pii()
        return self.masked_transcription

    def _propagate_pii(self):
        """Second pass to propagate highly specific context-dependent PII across the entire transcription."""
        propagate_entities = {
            "PARTIAL_NUMBER", "ACCOUNT_NUMBER", "PHONE_NUMBER",
            "CREDIT_CARD", "CREDIT_CARD_ENDING", "NATIONAL_ID",
            "EMAIL_ADDRESS", "PARENT_DETAILS"
        }
        cache = {}
        for _, mappings in self.pii_mappings:
            for placeholder, originals in mappings.items():
                entity = placeholder.strip("[]")
                if entity in propagate_entities:
                    for orig in originals:
                        cache[orig] = entity
                        
        if not cache:
            return
            
        # Sort by length descending to replace longer matches first
        cache = dict(sorted(cache.items(), key=lambda item: len(item[0]), reverse=True))

        for i, entry in enumerate(self.masked_transcription):
            text = entry["dialogue"]
            mappings = self.pii_mappings[i][1]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            
            for orig, entity in cache.items():
                escaped = re.escape(orig)
                if re.match(r'^\w+$', orig):
                    pattern = r'\b' + escaped + r'\b'
                else:
                    pattern = escaped
                
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for match in reversed(matches):
                        s, e = match.start(), match.end()
                        matched_str = text[s:e]
                        
                        placeholder = f"[{entity}]"
                        text = text[:s] + placeholder + text[e:]
                        mappings.setdefault(placeholder, []).append(matched_str)
                        
                        length = len(entry["dialogue"])
                        dur = end_time - start_time
                        if length > 0:
                            t0 = start_time + (s / length) * dur
                            t1 = start_time + (e / length) * dur
                        else:
                            t0, t1 = start_time, end_time
                        self.pii_timelines.append((t0, t1, matched_str, entity))
                        
            entry["dialogue"] = text
            self.pii_mappings[i] = (text, mappings)


    def unmask_transcription(self) -> List[Dict]:
        unmasked_transcription = []
        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            unmasked_text = masked_text
            for placeholder, original_values in mappings.items():
                for original_pii in original_values:
                    if placeholder in unmasked_text:
                        unmasked_text = unmasked_text.replace(placeholder, original_pii, 1)
            unmasked_entry = masked_entry.copy()
            unmasked_entry["dialogue"] = unmasked_text
            unmasked_transcription.append(unmasked_entry)
        return unmasked_transcription

    def get_pii_timelines(self) -> List[Tuple[float, float, str, str]]:
        return self.pii_timelines

    def create_fictitious_transcription(self) -> List[Dict]:
        fictitious_transcription = []
        available_surnames = list(self.surnames)
        name_mapping = {}

        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            # Clean up malformed placeholders (e.g., ERSON], [PERSON)
            fictitious_text = re.sub(r'\b(ERSON\]|\[PERSON)\b', '', masked_text)
            if fictitious_text != masked_text:
                print(f"Cleaned up malformed placeholders in text: {masked_text}")

            # Replace [PERSON] with consistent fictitious names
            if "[PERSON]" in mappings:
                for original_name in mappings["[PERSON]"]:
                    if "[PERSON]" in fictitious_text:
                        original_name_key = original_name.lower()
                        if original_name_key not in name_mapping:
                            first_name = random.choice(self.first_names)
                            surname = random.choice(available_surnames).capitalize()
                            name_mapping[original_name_key] = f"{first_name} {surname}"
                        fictitious_name = name_mapping[original_name_key]
                        fictitious_text = fictitious_text.replace("[PERSON]", fictitious_name, 1)

            # Replace [PARENT_DETAILS] with fictitious names
            if "[PARENT_DETAILS]" in mappings:
                for original_name in mappings["[PARENT_DETAILS]"]:
                    if "[PARENT_DETAILS]" in fictitious_text:
                        original_name_key = f"parent_{original_name.lower()}"
                        if original_name_key not in name_mapping:
                            first_name = random.choice(self.first_names)
                            surname = random.choice(available_surnames).capitalize()
                            name_mapping[original_name_key] = f"{first_name} {surname}"
                        fictitious_name = name_mapping[original_name_key]
                        fictitious_text = fictitious_text.replace("[PARENT_DETAILS]", fictitious_name, 1)

            # Replace [EMAIL_ADDRESS] with partially masked email
            if "[EMAIL_ADDRESS]" in mappings:
                for original_email in mappings["[EMAIL_ADDRESS]"]:
                    if "[EMAIL_ADDRESS]" in fictitious_text:
                        parts = original_email.split('@')
                        if len(parts) == 2:
                            local, domain = parts
                            domain_parts = domain.rsplit('.', 1)
                            if len(domain_parts) == 2:
                                domain_name, extension = domain_parts
                                masked_local = local[0] + '*' * (len(local) - 1)
                                masked_domain = domain_name[0] + '*' * (len(domain_name) - 1)
                                masked_email = f"{masked_local}@{masked_domain}.{extension}"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                            else:
                                masked_email = f"{local[0] + '*' * (len(local) - 1)}@*****"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                        else:
                            masked_email = "*****@*****"
                            fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)

            # Replace [PHONE_NUMBER] with masked version (keep last 4 digits)
            if "[PHONE_NUMBER]" in mappings:
                for original_phone in mappings["[PHONE_NUMBER]"]:
                    if "[PHONE_NUMBER]" in fictitious_text:
                        digits = re.sub(r'\D', '', original_phone)
                        if len(digits) >= 4:
                            masked_phone = '*' * (len(digits) - 4) + digits[-4:]
                        else:
                            masked_phone = '****'
                        fictitious_text = fictitious_text.replace("[PHONE_NUMBER]", masked_phone, 1)

            # Replace [ACCOUNT_NUMBER] with masked version
            if "[ACCOUNT_NUMBER]" in mappings:
                for original_acct in mappings["[ACCOUNT_NUMBER]"]:
                    if "[ACCOUNT_NUMBER]" in fictitious_text:
                        masked_acct = "XXXX-" + original_acct[-4:] if len(original_acct) >= 4 else "XXXX"
                        fictitious_text = fictitious_text.replace("[ACCOUNT_NUMBER]", masked_acct, 1)

            # Replace [CREDIT_CARD] with masked version
            if "[CREDIT_CARD]" in mappings:
                for original_card in mappings["[CREDIT_CARD]"]:
                    if "[CREDIT_CARD]" in fictitious_text:
                        digits = re.sub(r'\D', '', original_card)
                        if len(digits) >= 4:
                            masked_card = "XXXX-XXXX-XXXX-" + digits[-4:]
                        else:
                            masked_card = "XXXX-XXXX-XXXX-XXXX"
                        fictitious_text = fictitious_text.replace("[CREDIT_CARD]", masked_card, 1)

            # Replace [NATIONAL_ID] with masked version
            if "[NATIONAL_ID]" in mappings:
                for original_id in mappings["[NATIONAL_ID]"]:
                    if "[NATIONAL_ID]" in fictitious_text:
                        masked_id = "XX-XXXXXXX-X"
                        fictitious_text = fictitious_text.replace("[NATIONAL_ID]", masked_id, 1)

            # Replace [DATE_OF_BIRTH] with masked version
            if "[DATE_OF_BIRTH]" in mappings:
                for _ in mappings["[DATE_OF_BIRTH]"]:
                    if "[DATE_OF_BIRTH]" in fictitious_text:
                        fictitious_text = fictitious_text.replace("[DATE_OF_BIRTH]", "XX/XX/XXXX", 1)

            # [PHYSICAL_ADDRESS] — leave as placeholder (region names are generic enough)
            # No fictitious substitution needed for addresses

            fictitious_entry = masked_entry.copy()
            fictitious_entry["dialogue"] = fictitious_text
            fictitious_transcription.append(fictitious_entry)
        return fictitious_transcription

class PresidioPIIMasker:
    """
    PII masker for English-language call transcriptions using Microsoft Presidio.
    
    GDPR-aligned: Only detects entities that can trace back to the customer.
    Filters out: bank/company names, agent names, generic numbers, standalone dates.
    """

    # Deny-list: words that Presidio's PERSON detector often false-positives on.
    # These are bank names, company names, and common call center terms.
    PERSON_DENYLIST = {
        # Bank/financial institution names
        "bdo", "bpi", "metrobank", "landbank", "pnb", "rcbc", "unionbank",
        "chinabank", "eastwest", "securitybank", "maybank", "citi", "citibank",
        "hsbc", "deutsche", "barclays", "hdfc", "icici", "sbi", "axis",
        "kotak", "idfc", "bandhan", "mahindra", "bajaj", "tata", "reliance",
        "adani", "chase", "wells", "fargo", "standard", "chartered",
        # Company/org names
        "globe", "smart", "pldt", "meralco", "jollibee", "gcash", "paymaya",
        "lazada", "shopee", "grab", "uber",
        # Common false-positive words
        "good", "morning", "afternoon", "evening", "night", "fine", "thank", "okay",
        "customer", "agent", "representative", "supervisor", "manager",
        "sir", "madam", "maam", "miss", "mister",
        "collection", "recovery", "settlement", "finance", "payment",
        "account", "balance", "loan", "credit", "debit", "amount",
        "option", "press", "hold", "transfer", "connect",
        "angel", "may", "grace", "rose", "joy", "hope", "faith", "charity",
        "mark", "bill", "frank", "will", "art", "ray", "dale", "glen",
        "dean", "grant", "wade", "lance", "reed", "ford", "banks", "lane",
        "lake", "brook", "stone", "cash", "prince", "king", "queen",
        "summer", "winter", "spring", "dawn", "eve", "april", "june", "august",
    }

    # Words that should NEVER be masked, regardless of entity type.
    # Greeting/time-of-day words that Presidio sometimes classifies as DATE_TIME or PERSON.
    NON_PII_WORDS = {
        "morning", "afternoon", "evening", "night",
        "good morning", "good afternoon", "good evening", "good night",
        "today", "tomorrow", "yesterday",
        "hello", "hi", "bye", "goodbye",
        "thank you", "thanks", "okay", "yes", "no", "please", "sorry",
        "sir", "ma'am", "madam", "maam", "miss", "mister",
    }

    # Minimum confidence score for Presidio results
    MIN_SCORE = 0.6

    def __init__(self, regions_file: str = "philippines_regions.txt"):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        # List of common Filipino first names for fictitious name substitution
        self.first_names = [
            "Mark", "Anna", "Jose", "Maria", "John", "Clara", "Pedro", "Liza",
            "Ramon", "Teresa", "Ben", "Sofia", "Luis", "Emma", "Carlos"
        ]
        # Surnames and regions will be loaded dynamically if needed
        self.surnames = set()
        self.regions = self.load_regions(regions_file)

        # Custom recognizer for partial credit card numbers (context-aware)
        credit_card_ending_pattern = Pattern(
            name="credit_card_ending_pattern",
            regex=r"(?:credit|debit)\s*card\s*(?:ending|last\s*\d\s*digits?)\s*(?:is\s*|with\s*|in\s*)?\d{3,4}",
            score=0.85
        )
        credit_card_ending_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD_ENDING",
            patterns=[credit_card_ending_pattern]
        )
        self.analyzer.registry.add_recognizer(credit_card_ending_recognizer)

        # Custom recognizer for partial numbers (last 4 digits)
        partial_number_pattern = Pattern(
            name="partial_number_pattern",
            regex=r"(?i)(?:last\s*\d+\s*digits?|ending\s*(?:in|with))\s*(?:registered\s*with\s*[a-zA-Z]+\s*)?(?:is\s*|are\s*|with\s*|in\s*)?\d{3,4}",
            score=0.9
        )
        partial_number_recognizer = PatternRecognizer(
            supported_entity="PARTIAL_NUMBER",
            patterns=[partial_number_pattern]
        )
        self.analyzer.registry.add_recognizer(partial_number_recognizer)

        # Custom recognizer for contract reference numbers (CF-XXXX format only)
        contract_ref_pattern = Pattern(
            name="contract_ref_pattern",
            regex=r"\bCF[- ]?\d{4,}(?:\s*dash\s*\d+)?\b",
            score=0.95
        )
        contract_ref_recognizer = PatternRecognizer(
            supported_entity="CONTRACT_REF",
            patterns=[contract_ref_pattern]
        )
        self.analyzer.registry.add_recognizer(contract_ref_recognizer)

        # Custom recognizer for account numbers (context-required)
        account_number_pattern = Pattern(
            name="account_number_pattern",
            regex=r"(?:account|reference|loan|contract)\s*(?:number|no\.?)?\s*(?:is\s*)?\d{4,}(?:[- ]\d+)*",
            score=0.9
        )
        account_number_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[account_number_pattern]
        )
        self.analyzer.registry.add_recognizer(account_number_recognizer)

        # Custom recognizer for physical addresses (city/area names)
        if self.regions:
            physical_address_pattern = Pattern(
                name="physical_address_pattern",
                regex=r'\b(?:' + '|'.join(re.escape(region) for region in self.regions) + r')\b',
                score=0.95
            )
            physical_address_recognizer = PatternRecognizer(
                supported_entity="PHYSICAL_ADDRESS",
                patterns=[physical_address_pattern]
            )
            self.analyzer.registry.add_recognizer(physical_address_recognizer)

        # Custom recognizer for MONTH
        month_pattern = Pattern(
            name="month_pattern",
            regex=r"(?i)\b(?:January|February|March|April|May|June|July|August|"
                  r"September|October|November|December)\b",
            score=0.7
        )
        month_recognizer = PatternRecognizer(
            supported_entity="MONTH",
            patterns=[month_pattern]
        )
        self.analyzer.registry.add_recognizer(month_recognizer)

        # GDPR-relevant Presidio entities only.
        # Removed: CRYPTO, DATE_TIME, DOMAIN_NAME, NRP, URL, US_LICENSE_PLATE,
        #          MEDICAL_LICENSE, AU_ABN, AU_ACN, AU_TFN, AU_MEDICARE, UK_NHS,
        #          IP_ADDRESS (not relevant to call center PII)
        # Removed custom: FIGURE (causes massive over-matching of generic numbers)
        self.entities = [
            "CREDIT_CARD", "EMAIL_ADDRESS", "IBAN_CODE", "PERSON", "PHONE_NUMBER",
            "US_BANK_NUMBER", "US_ITIN", "US_PASSPORT", "US_SSN", "SG_NRIC_FIN",
            "CREDIT_CARD_ENDING", "CONTRACT_REF", "ACCOUNT_NUMBER", "PHYSICAL_ADDRESS",
            "MONTH", "PARTIAL_NUMBER"
        ]

    def load_surnames(self, surname_file: str) -> None:
        """Load surnames for fictitious name substitution if needed."""
        try:
            if os.path.exists(surname_file):
                with open(surname_file, "r", encoding="utf-8") as f:
                    self.surnames = {line.strip().lower() for line in f if line.strip()}
                print(f"Loaded {len(self.surnames)} surnames for fictitious names")
            else:
                print(f"Warning: Surname file {surname_file} not found for fictitious names.")
        except Exception as e:
            print(f"Error loading surnames for fictitious names: {e}")

    def load_regions(self, regions_file: str) -> set:
        """Load city/area names from a text file into a set for physical address matching."""
        regions = set()
        try:
            if os.path.exists(regions_file):
                with open(regions_file, "r", encoding="utf-8") as f:
                    regions = {line.strip() for line in f if line.strip()}
                print(f"Loaded {len(regions)} regions from {regions_file}")
            else:
                print(f"Warning: Regions file {regions_file} not found.")
        except Exception as e:
            print(f"Error loading regions: {e}")
        return regions

    def _filter_results(self, analyzer_results, text):
        """
        Post-analysis filter to remove false positives.
        - Removes PERSON detections that match the deny-list (bank names, common words)
        - Removes results below the minimum confidence score
        - Removes very short PERSON matches (likely false positives)
        """
        filtered = []
        for result in analyzer_results:
            # Skip low-confidence results
            if result.score < self.MIN_SCORE:
                continue

            original_text = text[result.start:result.end].strip()

            # Skip any result that matches a non-PII word (e.g. "morning", "afternoon")
            if original_text.lower() in self.NON_PII_WORDS:
                continue

            # For PERSON entity: check against deny-list and parent context
            if result.entity_type == "PERSON":
                # Only keep PERSON if preceded by parent/maiden name context
                before = text[max(0, result.start-40):result.start].lower()
                if not re.search(r'\b(mother|father|maiden|parent)\b', before):
                    continue
                # Skip if the detected text (or any word in it) is in the deny-list
                words = original_text.lower().split()
                if any(w in self.PERSON_DENYLIST for w in words):
                    continue
                # Skip very short matches (single character or 2-letter words)
                if len(original_text) <= 2:
                    continue
                    
            # For PHYSICAL_ADDRESS entity: skip false positives
            if result.entity_type == "PHYSICAL_ADDRESS":
                word_lower = original_text.lower()
                before = text[max(0, result.start-30):result.start].lower()
                
                # 1. Person name check (titles)
                if re.search(r'\b(mr|mrs|ms|miss|sir|maam|madam)\.?\s+(?:[a-z]+\s+){0,2}$', before):
                    continue
                    
                # 2. Require context for ALL single-word physical addresses
                # This prevents common words (Care, Making, Upon) and surnames (Maya, Madrid) from false matching
                if len(word_lower.split()) == 1:
                    location_context = r'\b(live|residing|address|city|municipality|brgy|barangay|in|at|from|to|street|st|ave|avenue|province|region)(?:\s+(?:is|of))?\s+$'
                    if not re.search(location_context, before):
                        continue
                        
                # 3. Collection Agency check (e.g. SP Madrid)
                if word_lower == "madrid" and re.search(r'\b(sp|s\.p\.?)\s+$', before):
                    continue
                    
            # For MONTH entity: only keep if near DOB context
            if result.entity_type == "MONTH":
                before = text[max(0, result.start-40):result.start].lower()
                if not re.search(r'\b(born|birth|dob|birthday|kapanganakan)\b', before):
                    continue
                    


            # Fix boundaries for custom patterns to only mask the digits, not the context phrase
            if result.entity_type in ("PARTIAL_NUMBER", "CREDIT_CARD_ENDING", "ACCOUNT_NUMBER"):
                m = re.search(r'\d{3,}(?:[- ]\d+)*$', original_text)
                if m:
                    result.start = result.start + m.start()

            filtered.append(result)
        return filtered

    def mask_transcription(self, transcription: List[Dict]) -> List[Dict]:
        """Mask PII in transcription, store mappings, and calculate PII timelines."""
        self.masked_transcription = []
        self.pii_mappings = []
        self.pii_timelines = []
        for entry in transcription:
            text = entry["dialogue"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            analyzer_results = self.analyzer.analyze(
                text=text, entities=self.entities, language="en",
                score_threshold=self.MIN_SCORE
            )
            # Apply post-analysis filtering for false positives
            analyzer_results = self._filter_results(analyzer_results, text)
            mappings = {}
            for result in analyzer_results:
                original_pii = text[result.start:result.end]
                placeholder = f"[{result.entity_type}]"
                mappings[placeholder] = mappings.get(placeholder, []) + [original_pii]
                text_length = len(text)
                if text_length > 0:
                    duration = end_time - start_time
                    pii_start_time = max(start_time, start_time + (result.start / text_length) * duration)
                    pii_end_time = min(end_time, start_time + (result.end / text_length) * duration)
                    self.pii_timelines.append((pii_start_time, pii_end_time, original_pii, result.entity_type))
            operators = {entity: OperatorConfig("replace", {"new_value": f"[{entity}]"}) for entity in self.entities}
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators,
            )
            masked_entry = entry.copy()
            masked_entry["dialogue"] = anonymized_result.text
            self.masked_transcription.append(masked_entry)
            self.pii_mappings.append((anonymized_result.text, mappings))
        
        self._propagate_pii()
        return self.masked_transcription

    def _propagate_pii(self):
        """Second pass to propagate highly specific context-dependent PII across the entire transcription."""
        propagate_entities = {
            "PARTIAL_NUMBER", "ACCOUNT_NUMBER", "PHONE_NUMBER",
            "CREDIT_CARD", "CREDIT_CARD_ENDING", "NATIONAL_ID",
            "EMAIL_ADDRESS", "PARENT_DETAILS"
        }
        cache = {}
        for _, mappings in self.pii_mappings:
            for placeholder, originals in mappings.items():
                entity = placeholder.strip("[]")
                if entity in propagate_entities:
                    for orig in originals:
                        cache[orig] = entity
                        
        if not cache:
            return
            
        # Sort by length descending to replace longer matches first
        cache = dict(sorted(cache.items(), key=lambda item: len(item[0]), reverse=True))

        for i, entry in enumerate(self.masked_transcription):
            text = entry["dialogue"]
            mappings = self.pii_mappings[i][1]
            start_time = entry["startTime"]
            end_time = entry["endTime"]
            
            for orig, entity in cache.items():
                escaped = re.escape(orig)
                if re.match(r'^\w+$', orig):
                    pattern = r'\b' + escaped + r'\b'
                else:
                    pattern = escaped
                
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for match in reversed(matches):
                        s, e = match.start(), match.end()
                        matched_str = text[s:e]
                        
                        placeholder = f"[{entity}]"
                        text = text[:s] + placeholder + text[e:]
                        mappings.setdefault(placeholder, []).append(matched_str)
                        
                        length = len(entry["dialogue"])
                        dur = end_time - start_time
                        if length > 0:
                            t0 = start_time + (s / length) * dur
                            t1 = start_time + (e / length) * dur
                        else:
                            t0, t1 = start_time, end_time
                        self.pii_timelines.append((t0, t1, matched_str, entity))
                        
            entry["dialogue"] = text
            self.pii_mappings[i] = (text, mappings)

    def unmask_transcription(self) -> List[Dict]:
        """Unmask PII using stored PII mappings."""
        unmasked_transcription = []
        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            unmasked_text = masked_text
            for placeholder, original_values in mappings.items():
                for original_pii in original_values:
                    if placeholder in unmasked_text:
                        unmasked_text = unmasked_text.replace(placeholder, original_pii, 1)
            unmasked_entry = masked_entry.copy()
            unmasked_entry["dialogue"] = unmasked_text
            unmasked_transcription.append(unmasked_entry)
        return unmasked_transcription

    def get_pii_timelines(self) -> List[Tuple[float, float, str, str]]:
        """Return timestamps of PII occurrences."""
        return self.pii_timelines

    def create_fictitious_transcription(self, surname_file: str) -> List[Dict]:
        """Create a transcription with PII placeholders replaced by fictitious/masked values."""
        if not self.surnames:
            self.load_surnames(surname_file)
        fictitious_transcription = []
        available_surnames = list(self.surnames)  # Convert set to list for random selection
        name_mapping = {}  # Dictionary to store original name to fictitious name mappings

        for masked_entry, (masked_text, mappings) in zip(self.masked_transcription, self.pii_mappings):
            fictitious_text = masked_text

            # Replace [PERSON] with consistent fictitious names
            if "[PERSON]" in mappings:
                for original_name in mappings["[PERSON]"]:
                    if "[PERSON]" in fictitious_text:
                        original_name_key = original_name.lower()
                        if original_name_key not in name_mapping:
                            first_name = random.choice(self.first_names)
                            surname = random.choice(available_surnames).capitalize()
                            name_mapping[original_name_key] = f"{first_name} {surname}"
                        fictitious_name = name_mapping[original_name_key]
                        fictitious_text = fictitious_text.replace("[PERSON]", fictitious_name, 1)

            # Replace [EMAIL_ADDRESS] with partially masked email
            if "[EMAIL_ADDRESS]" in mappings:
                for original_email in mappings["[EMAIL_ADDRESS]"]:
                    if "[EMAIL_ADDRESS]" in fictitious_text:
                        parts = original_email.split('@')
                        if len(parts) == 2:
                            local, domain = parts
                            domain_parts = domain.rsplit('.', 1)
                            if len(domain_parts) == 2:
                                domain_name, extension = domain_parts
                                masked_local = local[0] + '*' * (len(local) - 1)
                                masked_domain = domain_name[0] + '*' * (len(domain_name) - 1)
                                masked_email = f"{masked_local}@{masked_domain}.{extension}"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                            else:
                                masked_email = f"{local[0] + '*' * (len(local) - 1)}@*****"
                                fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)
                        else:
                            masked_email = "*****@*****"
                            fictitious_text = fictitious_text.replace("[EMAIL_ADDRESS]", masked_email, 1)

            # Replace [PHONE_NUMBER] with masked version (keep last 4 digits)
            if "[PHONE_NUMBER]" in mappings:
                for original_phone in mappings["[PHONE_NUMBER]"]:
                    if "[PHONE_NUMBER]" in fictitious_text:
                        digits = re.sub(r'\D', '', original_phone)
                        if len(digits) >= 4:
                            masked_phone = '*' * (len(digits) - 4) + digits[-4:]
                        else:
                            masked_phone = '****'
                        fictitious_text = fictitious_text.replace("[PHONE_NUMBER]", masked_phone, 1)

            # Replace [CREDIT_CARD] with masked version
            if "[CREDIT_CARD]" in mappings:
                for original_card in mappings["[CREDIT_CARD]"]:
                    if "[CREDIT_CARD]" in fictitious_text:
                        digits = re.sub(r'\D', '', original_card)
                        if len(digits) >= 4:
                            masked_card = "XXXX-XXXX-XXXX-" + digits[-4:]
                        else:
                            masked_card = "XXXX-XXXX-XXXX-XXXX"
                        fictitious_text = fictitious_text.replace("[CREDIT_CARD]", masked_card, 1)

            # Replace [CREDIT_CARD_ENDING] with masked version
            if "[CREDIT_CARD_ENDING]" in mappings:
                for _ in mappings["[CREDIT_CARD_ENDING]"]:
                    if "[CREDIT_CARD_ENDING]" in fictitious_text:
                        fictitious_text = fictitious_text.replace("[CREDIT_CARD_ENDING]", "card ending XXXX", 1)

            # Replace [ACCOUNT_NUMBER] / [CONTRACT_REF] with masked version
            for tag in ("[ACCOUNT_NUMBER]", "[CONTRACT_REF]"):
                if tag in mappings:
                    for original_val in mappings[tag]:
                        if tag in fictitious_text:
                            masked_val = "XXXX-" + original_val[-4:] if len(original_val) >= 4 else "XXXX"
                            fictitious_text = fictitious_text.replace(tag, masked_val, 1)

            # Replace any remaining ID-type placeholders
            for tag in ("[US_SSN]", "[US_ITIN]", "[US_PASSPORT]", "[SG_NRIC_FIN]",
                        "[IBAN_CODE]", "[US_BANK_NUMBER]"):
                if tag in mappings:
                    for _ in mappings[tag]:
                        if tag in fictitious_text:
                            fictitious_text = fictitious_text.replace(tag, "XX-XXXXXXX-X", 1)

            fictitious_entry = masked_entry.copy()
            fictitious_entry["dialogue"] = fictitious_text
            fictitious_transcription.append(fictitious_entry)
        return fictitious_transcription

class PydubPIIMasker:
    """PII audio masker using pydub."""
    
    def mask_audio(self, audio_path: str, beep_path: str, pii_timelines: List[Tuple[float, float, str, str]]) -> str:
        """Mask PII segments in audio using a beep sound using pydub."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep file not found: {beep_path}")
            
        print(f"Loading audio file: {audio_path}")
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e}. Ensure it's a valid MP3/WAV file.")
                
        print(f"Loading beep file: {beep_path}")
        beep = AudioSegment.from_wav(beep_path)
        
        pii_timelines_sorted = sorted(pii_timelines, key=lambda x: x[0])
        
        print(f"Processing {len(pii_timelines_sorted)} PII occurrences...")
        
        processed_segments = []
        last_end = 0
        
        for i, (start_time, end_time, text, entity_type) in enumerate(pii_timelines_sorted):
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            print(f"Processing PII segment {i+1}/{len(pii_timelines_sorted)}: {entity_type} - '{text}'")
            print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s ({start_ms}ms - {end_ms}ms)")
            
            if start_ms > last_end:
                print(f"  Adding segment from {last_end}ms to {start_ms}ms")
                segment = audio[last_end:start_ms]
                processed_segments.append(segment)
            
            duration_ms = end_ms - start_ms
            
            if len(beep) < duration_ms:
                print(f"  Extending beep to match PII duration: {duration_ms}ms")
                beep_segment = beep * (int(duration_ms / len(beep)) + 1)
                beep_segment = beep_segment[:duration_ms]
            else:
                print(f"  Trimming beep to match PII duration: {duration_ms}ms")
                beep_segment = beep[:duration_ms]
            
            segment_volume = audio[start_ms:end_ms].dBFS
            if segment_volume > -float('inf'):
                print(f"  Matching volume: {segment_volume:.2f} dBFS")
                beep_segment = beep_segment.apply_gain(segment_volume - beep_segment.dBFS)
            
            processed_segments.append(beep_segment)
            
            last_end = end_ms
        
        if last_end < len(audio):
            print(f"  Adding final segment from {last_end}ms to end")
            processed_segments.append(audio[last_end:])
        
        print("Combining all audio segments...")
        masked_audio = sum(processed_segments)
        
        output_path = audio_path.replace('.wav', '_masked.wav')
        print(f"Exporting masked audio to: {output_path}")
        masked_audio.export(output_path, format="wav")
        
        return output_path

def main():
    # Input files (example paths, can be modified)
    transcription_path = "/home/ashutosh/Downloads/Transcription - 4344"
    audio_path = "/home/ashutosh/Downloads/87b3ced0-c6dc-428e-aee2-abd07e2b2b2b_predictive-09187515730-882000067-20250207-143503-1738910070.0722816_615135344344.wav"
    beep_path = "beep.wav"
    surname_file = "allnamesnew.txt"
    regions_file = "PII_masking/philippines_regions.txt"

    # Generate beep sound
    try:
        generate_beep_wave(beep_path, frequency=1000.0, duration=1.0, sample_rate=44100, amplitude=0.8)
        print(f"Generated beep sound: {beep_path}")
    except Exception as e:
        print(f"Error generating beep.wav: {e}")
        return

    # Load transcription
    try:
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription = json.load(f)
    except FileNotFoundError:
        print(f"Error: {transcription_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {transcription_path}.")
        return

    # Detect language
    language, filipino_percentage = detect_language(transcription, surname_file)
    print(f"Detected language: {language}")

    # Initialize appropriate masker
    if language == "filipino":
        masker = FilipinoPIIMasker(surname_file=surname_file, regions_file=regions_file)
        masked_output = "filipino_masked_transcription.json"
        unmasked_output = "filipino_unmasked_transcription.json"
    else:
        masker = PresidioPIIMasker(regions_file=regions_file)
        masked_output = "presidio_masked_transcription.json"
        unmasked_output = "presidio_unmasked_transcription.json"

    # Mask transcription and get PII timelines
    try:
        # First call mask_transcription to populate pii_mappings and pii_timelines
        _ = masker.mask_transcription(transcription)
        # Use fictitious transcription for masked output
        masked_transcription = masker.create_fictitious_transcription() if language == "filipino" else masker.create_fictitious_transcription(surname_file)
        with open(masked_output, "w", encoding="utf-8") as f:
            json.dump(masked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error masking transcription: {e}")
        return

    # Unmask transcription
    try:
        unmasked_transcription = masker.unmask_transcription()
        with open(unmasked_output, "w", encoding="utf-8") as f:
            json.dump(unmasked_transcription, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error unmasking transcription: {e}")
        return

    # Get PII timelines
    pii_timelines = masker.get_pii_timelines()

    # Initialize Pydub masker and mask audio
    pydub_masker = PydubPIIMasker()
    try:
        output_path = pydub_masker.mask_audio(audio_path, beep_path, pii_timelines)
        print(f"Successfully masked audio file: {output_path}")
    except Exception as e:
        print(f"Error processing audio with Pydub: {e}")
        return

    # Print results
    print("\nMasked Transcription:")
    print(json.dumps(masked_transcription, indent=2, ensure_ascii=False))
    # print("\nUnmasked Transcription:")
    # print(json.dumps(unmasked_transcription, indent=2, ensure_ascii=False))
    # print("\nPII Timelines (start, end, text, entity_type):")
    # print(json.dumps(pii_timelines, indent=2, ensure_ascii=False))
    
    # print(f"\nSuccessfully processed {len(pii_timelines)} PII occurrences in the audio.")
    # print(f"Masked audio saved as: {output_path}")
    # print(f"Detected language: {language}")
    # print(f"Language percentage: {filipino_percentage}")

if __name__ == "__main__":
    main()
