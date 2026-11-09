"""
SCP-ECG Diagnostic Code Mapper for Multi-Territory Myocardial Infarction.
Maps PTB-XL diagnostic statements to 5 anatomical target classes according to
the ESC/ACC/AHA/WHF Fourth Universal Definition of Myocardial Infarction (2018).
"""

import ast
from typing import Dict, List, Tuple

TARGET_CLASSES = [
    "NORM",  # 0: Normal Sinus Rhythm
    "AMI",   # 1: Anterior Myocardial Infarction (LAD: V1-V4)
    "IMI",   # 2: Inferior Myocardial Infarction (RCA: II, III, aVF)
    "LMI",   # 3: Lateral Myocardial Infarction (LCx: I, aVL, V5, V6)
    "SMI"    # 4: Septal Myocardial Infarction (LAD Septal branch: V1-V2)
]

NUM_CLASSES = len(TARGET_CLASSES)

# Medical ontology mapping based on AHA/ESC standard statements
SCP_TERRITORY_RULES = {
    "AMI": ["AMI", "ASMI", "ALMI", "INJAS", "INJAL"],
    "IMI": ["IMI", "ILMI", "IPLMI", "IPMI", "INJIN", "INJIL", "PMI"],
    "LMI": ["LMI", "ILMI", "ALMI", "IPLMI", "INJAL", "INJLA", "INJIL"],
    "SMI": ["ASMI", "INJAS", "SMI"]
}


class SCPMapper:
    """
    Parses SCP-ECG statements string from PTB-XL metadata and converts into
    a multi-label binary ground truth vector of length 5:
    [y_NORM, y_AMI, y_IMI, y_LMI, y_SMI]
    """

    @staticmethod
    def parse_scp_string(scp_str: str) -> Dict[str, float]:
        """
        Parses dict string such as "{'NORM': 100.0, 'LVOLT': 0.0}".
        """
        if not scp_str or scp_str == "nan":
            return {}
        try:
            return ast.literal_eval(scp_str)
        except (ValueError, SyntaxError):
            return {}

    @classmethod
    def get_multilabel_vector(cls, scp_str: str) -> List[float]:
        """
        Returns binary float vector [y_0, y_1, y_2, y_3, y_4].
        """
        stmts = cls.parse_scp_string(scp_str)
        labels = [0.0] * NUM_CLASSES

        # 0. Normal
        if "NORM" in stmts:
            labels[0] = 1.0

        # 1. Anterior MI (LAD)
        if any(code in stmts for code in SCP_TERRITORY_RULES["AMI"]):
            labels[1] = 1.0

        # 2. Inferior MI (RCA)
        if any(code in stmts for code in SCP_TERRITORY_RULES["IMI"]):
            labels[2] = 1.0

        # 3. Lateral MI (LCx)
        if any(code in stmts for code in SCP_TERRITORY_RULES["LMI"]):
            labels[3] = 1.0

        # 4. Septal MI (LAD Septal)
        if any(code in stmts for code in SCP_TERRITORY_RULES["SMI"]):
            labels[4] = 1.0

        return labels

    @classmethod
    def decode_predictions(cls, probs: List[float], threshold: float = 0.5) -> List[str]:
        """
        Decodes output probabilities into clinical textual diagnoses.
        """
        diagnoses = []
        for i, p in enumerate(probs):
            if p >= threshold:
                diagnoses.append(f"{TARGET_CLASSES[i]} ({p*100:.1f}%)")
        return diagnoses if diagnoses else ["UNDETERMINED / OTHER (<50%)"]
