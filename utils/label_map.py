"""
Mapping between disease labels and Khmer/English names
"""
from utils.mapping import KHMER_LABELS, DISPLAY_LABELS

# Disease information with Khmer and English names, advice, and sources
DISEASE_INFO = {
    "bacterial_leaf_blight": {
        "km": {
            "name": "ជម្ងឺខ្លាញ់ស្លឹកបាក់តេរី",
            "advice": [
                "ប្រើពូជធន់នឹងជំងឺ (IRRI/MAFF).",
                "កុំប្រើជីអាសូតច្រើនពេក។",
                "ធ្វើការបង្ហូរទឹកឱ្យល្អ។",
                "លុបស្លឹកឆ្លងខ្លាំង។",
                "បន្ថែមជីប៉ូតាស្យូម។",
                "អនុវត្តការបង្វិលដំណាំដើម្បីកាត់បន្ថយមេរោគ។"
            ],
            "sources": {
                "IRRI – Bacterial Blight": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/bacterial-blight",
                "FAO – BLB Management": "https://www.fao.org/3/y5704e/y5704e04.htm",
                "MAFF Cambodia – Rice Diseases": "http://www.maff.gov.kh/",
            }
        },
        "en": {
            "name": "Bacterial Leaf Blight (BLB)",
            "advice": [
                "Use resistant or tolerant varieties.",
                "Avoid excessive nitrogen.",
                "Improve drainage to reduce standing water.",
                "Remove heavily diseased tillers.",
                "Apply potassium fertilizer.",
                "Practice crop rotation."
            ],
            "sources": {
                "IRRI – Bacterial Blight": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/bacterial-blight",
                "FAO – IPM Guide": "https://www.fao.org/3/y5704e/y5704e04.htm",
                "GDA Cambodia": "https://gda.maff.gov.kh/",
            }
        },
    },
    "brown_spot": {
        "km": {
            "name": "ជម្ងឺចំណុចត្នោត",
            "advice": [
                "បន្ថែមជីប៉ូតាស្យូម និងជីសរីរាង្គ។",
                "ប្រើគ្រាប់ពូជមានគុណភាពល្អ។",
                "រក្សាសំណើមដីឱ្យសមរម្យ។",
                "រក្សាចន្លោះដាំឱ្យសមរម្យ។",
                "ធ្វើការកែលម្អដីតាម MAFF។"
            ],
            "sources": {
                "IRRI – Brown Spot": "https://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/brown-spot",
                "FAO – Nutrient Management": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "MAFF Cambodia": "http://www.maff.gov.kh/",
            }
        },
        "en": {
            "name": "Brown Spot",
            "advice": [
                "Apply potassium and organic fertilizers.",
                "Maintain adequate soil moisture.",
                "Use high-quality seeds.",
                "Maintain proper spacing.",
                "Improve soil fertility as recommended."
            ],
            "sources": {
                "IRRI – Brown Spot": "https://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/brown-spot",
                "FAO – Rice IPM": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "GDA Cambodia": "https://gda.maff.gov.kh/",
            }
        },
    },
    "leaf_blast": {
        "km": {
            "name": "ជម្ងឺផ្លេះស្លឹក",
            "advice": [
                "ប្រើពូជធន់នឹង Blast.",
                "ចែកចាយជីអាសូតជាបន្តបន្ទាប់។",
                "រក្សាចន្លោះដាំឱ្យមានខ្យល់ចេញចូល។",
                "ធ្វើការបង្ហូរទឹកឱ្យល្អ។",
                "បន្ថែមជី K ដើម្បីពង្រឹងភាពធន់។"
            ],
            "sources": {
                "IRRI – Rice Blast": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/rice-blast",
                "FAO – Blast Management": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "GDA Cambodia": "https://gda.maff.gov.kh/",
            }
        },
        "en": {
            "name": "Leaf Blast",
            "advice": [
                "Grow blast-resistant varieties.",
                "Split nitrogen application; avoid heavy early doses.",
                "Improve airflow with proper spacing.",
                "Keep fields well-drained.",
                "Apply potassium to strengthen plants."
            ],
            "sources": {
                "IRRI – Rice Blast": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/rice-blast",
                "FAO – Rice Diseases": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "MAFF – Plant Protection": "http://www.maff.gov.kh/",
            }
        },
    },
    "leaf_scald": {
        "km": {
            "name": "ជម្ងឺដុតស្លឹក",
            "advice": [
                "គ្រប់គ្រងទឹកឱ្យសមរម្យ។",
                "កាត់បន្ថយជីអាសូត។",
                "លុបស្លឹកឆ្លងខ្លាំង។",
                "ប្រើ N-P-K តាមស្តង់ដារ MAFF។",
                "ចៀសវាងរបួសដំណាំ។"
            ],
            "sources": {
                "IRRI – Leaf Scald": "https://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/leaf-scald",
                "FAO – IPM Manual": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "MAFF Cambodia": "http://www.maff.gov.kh/"
            }
        },
        "en": {
            "name": "Leaf Scald",
            "advice": [
                "Manage water properly; avoid stagnant water.",
                "Reduce nitrogen if excessive.",
                "Remove severely infected leaves.",
                "Follow MAFF N-P-K fertilizer guidelines.",
                "Avoid mechanical injury."
            ],
            "sources": {
                "IRRI – Leaf Scald": "https://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/leaf-scald",
                "FAO – IPM Manual": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "GDA Cambodia": "https://gda.maff.gov.kh/"
            }
        },
    },
    "sheath_blight": {
        "km": {
            "name": "ជម្ងឺខ្លាញ់កណ្ដាលដើម",
            "advice": [
                "កុំដាំចង្អៀតពេក។",
                "គ្រប់គ្រងជីអាសូតឱ្យសមរម្យ។",
                "រក្សាខ្យល់ចេញចូលក្នុងស្រែ។",
                "កាត់ដើមឆ្លងដំបូងៗ។",
                "គ្រប់គ្រងសំណល់ដំណាំ (FAO)."
            ],
            "sources": {
                "IRRI – Sheath Blight": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/sheath-blight",
                "FAO – Residue Management": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "MAFF Cambodia": "http://www.maff.gov.kh/"
            }
        },
        "en": {
            "name": "Sheath Blight",
            "advice": [
                "Avoid dense planting.",
                "Apply nitrogen carefully.",
                "Improve canopy ventilation.",
                "Remove infected tillers early.",
                "Follow FAO crop residue management guidelines."
            ],
            "sources": {
                "IRRI – Sheath Blight": "https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/sheath-blight",
                "FAO – IPM Guide": "https://www.fao.org/3/y0700e/y0700e04.htm",
                "GDA Cambodia": "https://gda.maff.gov.kh/"
            }
        },
    },
    "healthy": {
        "km": {
            "name": "ស្រូវមានសុខភាពល្អ",
            "advice": [
                "រក្សាទឹក និងជីឱ្យសមរម្យ។",
                "តាមដានស្រែជាប្រចាំ។",
                "ប្រើគ្រាប់ពូជដែលបានផ្ទៀងផ្ទាត់ពី MAFF/IRRI។"
            ],
            "sources": {
                "IRRI – Healthy Crop Guide": "https://www.knowledgebank.irri.org",
                "MAFF Cambodia": "http://www.maff.gov.kh/"
            }
        },
        "en": {
            "name": "Healthy Leaf",
            "advice": [
                "Maintain balanced water and nutrients.",
                "Monitor crop conditions regularly.",
                "Use certified seeds approved by MAFF/IRRI."
            ],
            "sources": {
                "IRRI – Crop Manager": "https://www.knowledgebank.irri.org",
                "GDA Cambodia": "https://gda.maff.gov.kh/",
            }
        },
    },
}

# Label to Khmer character mapping (for consistency with reference structure)
LABEL_TO_KHMER = KHMER_LABELS

# Label to display name mapping
LABEL_TO_DISPLAY = DISPLAY_LABELS
