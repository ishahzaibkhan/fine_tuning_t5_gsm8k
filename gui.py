import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit,
    QVBoxLayout, QWidget
)
from PyQt5.QtGui import QFont
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model():
    try:
        model = T5ForConditionalGeneration.from_pretrained("./t5_gsm8k_model")
        tokenizer = T5Tokenizer.from_pretrained("./t5_gsm8k_model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def solve_math_problem(model_tokenizer_device, problem):
    model, tokenizer, device = model_tokenizer_device
    try:
        input_text = "question: " + problem
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=128, num_beams=3)
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return solution
    except Exception as e:
        return f"Error during inference: {e}"

class MathSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fine-tuning T5 Model for Grade School Math Problem Solving")
        self.setGeometry(100, 100, 700, 500)

        # Load the model
        self.model = load_model()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(14)

        self.input_label = QLabel("Enter a math problem:")
        self.input_label.setFont(font)
        layout.addWidget(self.input_label)

        self.input_box = QLineEdit()
        self.input_box.setFont(font)
        self.input_box.setMinimumHeight(40)
        layout.addWidget(self.input_box)

        self.solve_button = QPushButton("Solve")
        self.solve_button.setFont(font)
        self.solve_button.setMinimumHeight(45)
        self.solve_button.clicked.connect(self.solve_problem)
        layout.addWidget(self.solve_button)

        self.output_label = QLabel("Solution:")
        self.output_label.setFont(font)
        layout.addWidget(self.output_label)

        self.output_box = QTextEdit()
        self.output_box.setFont(font)
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(120)
        layout.addWidget(self.output_box)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def solve_problem(self):
        problem = self.input_box.text()
        if problem.strip():
            solution = solve_math_problem(self.model, problem)
            self.output_box.setText(solution)
        else:
            self.output_box.setText("Please enter a valid math problem.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MathSolverGUI()
    window.show()
    sys.exit(app.exec_())