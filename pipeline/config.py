instrument_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
verb_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
target_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
phase_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
vqa_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
mrg_agent_system_prompt = "You are an AI assistant specializing in surgical video analysis."
mrg_refine_agent_system_prompt = """You are an AI assistant specializing in surgical video analysis.
You can also imagine yourself as a lecturer on surgery who explains surgeons' thought processes and other surgical rationales to new junior surgeons who ask you questions.
Your task is to provide detailed, medically accurate, and contextually relevant responses for surgical video frame analysis questions."""

mcq_output_format_constraint = """You are answering a multiple-choice question.
There may be one or more correct options.
Valid option letters: [{letter_headers}].
Output only the correct option letters, concatenated with no spaces or punctuation.
Do not include any explanation, punctuation, or additional text."""
phase_output_format_constraint = """You are answering a multiple-choice question. 
Only output the letter of the correct option ({letter_headers}). 
Do not include any explanation, punctuation, or additional text."""
open_description_format_constraint = """Important: 1. Respond in English only. 2. Do not use Markdown formatting. 3. Do not include code blocks, lists, or special symbols."""
mrg_format_constraint = """Important: 1. Respond in English only. 2. Do not use Markdown formatting. 3. Do not include code blocks, lists, or special symbols."""
mrg_task_prompt = """Based on the visual content and this clinical context, please generate a detailed medical report addressing the following: {question}

Please provide a comprehensive medical report that includes:
1. Observation of visible findings, surgical progress, and instrument usage.
2. Clinical significance of this moment/phase.
3. Relevant clinical considerations or next procedural steps."""
MRG_REFINE_TEMPLATE = """User:
Please summarize the following medical report in three labeled lines:
Observation: [concise description of visible findings, progress, and instruments]
Clinical Significance: [explain clinical meaning or purpose of this phase]
Next Steps: [likely upcoming actions or considerations]
Avoid repetition and ensure medical precision. Please provide your answer in a clear, text-based format, but do not use any markdown syntax.

Medical Report:
{context}
"""

instrument_grounding_model = "gemini-2.5-flash"
instrument_identification_model = "gemini-2.5-flash"
verb_identification_model = "gemini-2.5-flash"
target_identification_model = "gemini-2.5-flash"
phase_classification_model = "gemini-2.5-flash"
open_description_model = "gemini-2.5-flash"
# mrg_model = "Qwen/Qwen3-32B"
mrg_model = "gemini-2.5-flash"
mrg_refine_model = "Qwen/Qwen3-32B"

triplet_mapping = [
    "grasper,dissect,cystic_plate",
    "grasper,dissect,gallbladder",
    "grasper,dissect,omentum",
    "grasper,grasp,cystic_artery",
    "grasper,grasp,cystic_duct",
    "grasper,grasp,cystic_pedicle",
    "grasper,grasp,cystic_plate",
    "grasper,grasp,gallbladder",
    "grasper,grasp,gut",
    "grasper,grasp,liver",
    "grasper,grasp,omentum",
    "grasper,grasp,peritoneum",
    "grasper,grasp,specimen_bag",
    "grasper,pack,gallbladder",
    "grasper,retract,cystic_duct",
    "grasper,retract,cystic_pedicle",
    "grasper,retract,cystic_plate",
    "grasper,retract,gallbladder",
    "grasper,retract,gut",
    "grasper,retract,liver",
    "grasper,retract,omentum",
    "grasper,retract,peritoneum",
    "bipolar,coagulate,abdominal_wall_cavity",
    "bipolar,coagulate,blood_vessel",
    "bipolar,coagulate,cystic_artery",
    "bipolar,coagulate,cystic_duct",
    "bipolar,coagulate,cystic_pedicle",
    "bipolar,coagulate,cystic_plate",
    "bipolar,coagulate,gallbladder",
    "bipolar,coagulate,liver",
    "bipolar,coagulate,omentum",
    "bipolar,coagulate,peritoneum",
    "bipolar,dissect,adhesion",
    "bipolar,dissect,cystic_artery",
    "bipolar,dissect,cystic_duct",
    "bipolar,dissect,cystic_plate",
    "bipolar,dissect,gallbladder",
    "bipolar,dissect,omentum",
    "bipolar,grasp,cystic_plate",
    "bipolar,grasp,liver",
    "bipolar,grasp,specimen_bag",
    "bipolar,retract,cystic_duct",
    "bipolar,retract,cystic_pedicle",
    "bipolar,retract,gallbladder",
    "bipolar,retract,liver",
    "bipolar,retract,omentum",
    "hook,coagulate,blood_vessel",
    "hook,coagulate,cystic_artery",
    "hook,coagulate,cystic_duct",
    "hook,coagulate,cystic_pedicle",
    "hook,coagulate,cystic_plate",
    "hook,coagulate,gallbladder",
    "hook,coagulate,liver",
    "hook,coagulate,omentum",
    "hook,cut,blood_vessel",
    "hook,cut,peritoneum",
    "hook,dissect,blood_vessel",
    "hook,dissect,cystic_artery",
    "hook,dissect,cystic_duct",
    "hook,dissect,cystic_plate",
    "hook,dissect,gallbladder",
    "hook,dissect,omentum",
    "hook,dissect,peritoneum",
    "hook,retract,gallbladder",
    "hook,retract,liver",
    "scissors,coagulate,omentum",
    "scissors,cut,adhesion",
    "scissors,cut,blood_vessel",
    "scissors,cut,cystic_artery",
    "scissors,cut,cystic_duct",
    "scissors,cut,cystic_plate",
    "scissors,cut,liver",
    "scissors,cut,omentum",
    "scissors,cut,peritoneum",
    "scissors,dissect,cystic_plate",
    "scissors,dissect,gallbladder",
    "scissors,dissect,omentum",
    "clipper,clip,blood_vessel",
    "clipper,clip,cystic_artery",
    "clipper,clip,cystic_duct",
    "clipper,clip,cystic_pedicle",
    "clipper,clip,cystic_plate",
    "irrigator,aspirate,fluid",
    "irrigator,dissect,cystic_duct",
    "irrigator,dissect,cystic_pedicle",
    "irrigator,dissect,cystic_plate",
    "irrigator,dissect,gallbladder",
    "irrigator,dissect,omentum",
    "irrigator,irrigate,abdominal_wall_cavity",
    "irrigator,irrigate,cystic_pedicle",
    "irrigator,irrigate,liver",
    "irrigator,retract,gallbladder",
    "irrigator,retract,liver",
    "irrigator,retract,omentum",
]
open_description_questions = [
    "Can you describe the findings visible in this endoscopy image?",
    "What can you observe in this endoscopic image?",
    "Can you provide a detailed description of what's visible in this endoscopy image?",
    "Please describe the characteristics of the tissue shown in this endoscopic image.",
    "What are the key features you can identify in this endoscopy image?",
    "Can you analyze the visual features of this endoscopy image?",
    "What abnormalities or normal findings can you identify in this image?",
    "Please provide an analysis of the findings in this endoscopy image.",
    "Can you describe the characteristics of any lesions or tissue in this endoscopy image?",
    "What diagnostic features are visible in this endoscopic visualization?",
]
phase_descriptions = [
    "the preparation phase",
    "Calot's triangle dissection phase",
    "cystic duct clipping and cutting phase",
    "gallbladder dissection phase",
    "gallbladder packaging or detachment phase",
    "the cleaning and coagulation phase",
    "gallbladder retraction phase",
]
instrument_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
