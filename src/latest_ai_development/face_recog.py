# src/your_project/face_verification_crew.py

import os
import cv2
import numpy as np
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ------------------------------
# ✅ CrewAI Tool: Face Matching Tool
# ------------------------------
@tool("face_match_tool")
def face_match_tool(image1_path: str, image2_path: str) -> str:
    """
    Compares two face images and determines if they likely belong to the same person.
    It returns a similarity score and interpretation.
    """

    def detect_and_crop_face(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return None

        (x, y, w, h) = faces[0]
        cropped_face = gray[y:y+h, x:x+w]
        return cropped_face

    def compare_faces(face1, face2):
        face1_resized = cv2.resize(face1, (100, 100))
        face2_resized = cv2.resize(face2, (100, 100))
        difference = cv2.absdiff(face1_resized, face2_resized)
        score = np.mean(difference)
        return score

    face1 = detect_and_crop_face(image1_path)
    face2 = detect_and_crop_face(image2_path)

    if face1 is None or face2 is None:
        return "❌ Face detection failed for one or both images."

    difference_score = compare_faces(face1, face2)

    if difference_score < 50:
        result = (
            f"✅ Face difference score: {difference_score:.2f}. "
            "The faces likely belong to the same person."
        )
    else:
        result = (
            f"⚠️ Face difference score: {difference_score:.2f}. "
            "The faces do not belong to the same person."
        )

    return result

# ------------------------------
# ✅ CrewAI Agent: Face Verification Expert
# ------------------------------
face_verifier_agent = Agent(
    role="Face Verification Expert",
    goal="Analyze face photos and determine if they are of the same person.",
    backstory=(
        "You are a skilled forensic image analyst trained to verify identity "
        "through facial similarity analysis. You assess visual inputs and return clear verdicts."
    ),
    verbose=True,
    memory=True,
    tools=[face_match_tool]
)

# ------------------------------
# ✅ CrewAI Task: Compare Two Faces
# ------------------------------
face_match_task = Task(
    description=(
        "You are given two image paths: {image1_path} and {image2_path}. "
        "Your job is to analyze the faces and determine if they belong to the same person. "
        "Provide a score and a brief analysis in your final report."
    ),
    expected_output="A verification report including face difference score and similarity conclusion.",
    agent=face_verifier_agent
)

# ------------------------------
# ✅ CrewAI Execution: Running the Face Match Workflow
# ------------------------------
crew = Crew(
    agents=[face_verifier_agent],
    tasks=[face_match_task],
    process=Process.sequential
)

# ------------------------------
# ✅ Run the Task with Example Face Images
# ------------------------------
if __name__ == "__main__":
    image1 = "src\latest_ai_development\pallavi1.jpg"
    image2 = "src\latest_ai_development\pallavi2.jpeg"

    inputs = {"image1_path": image1, "image2_path": image2}
    result = crew.kickoff(inputs=inputs)

    print("\n🧠 Face Verification Result:\n", result)
