# Math Problem Solver Using Fine-Tuned T5 Model

This project demonstrates a GUI-based math problem solver using a fine-tuned T5 model (`t5-small`) trained on the GSM8K dataset. The model is trained to solve grade school math word problems through step-by-step reasoning.

---

## Features
- **GUI Interface**: Built with PyQt5 for easy interaction.
- **Fine-Tuned Model**: Uses a T5-small model fine-tuned on GSM8K for solving math problems.
- **Inference**: Provides step-by-step reasoning for math problems.
- **Prefix Support**: Uses the `"solve: "` prefix for training and inference.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch
- Hugging Face Transformers
- PyQt5

### Install Dependencies
Run the following commands to install required libraries:
```bash
pip install torch transformers sentencepiece protobuf pyqt5
```

---

## Usage

### Run the GUI
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run the Python script:
   ```bash
   python main.py
   ```
3. Enter a math problem in the input box and click "Solve" to get the solution.

---

## Training Details

### Dataset
- **Source**: GSM8K (Grade School Math Problems)
- **Training Prefix**: `"solve: "` added to each question.

### Model
- **Base Model**: `t5-small` (from Hugging Face Transformers)
- **Fine-Tuning**:
  - **Epochs**: 20
  - **Batch Size**: 8
  - **Learning Rate**: 5e-5
  - **Max Input Length**: 256
  - **Max Target Length**: 256

### Saving
- The fine-tuned model is saved in the `t5_gsm8k_model/` folder.

---

## Example Questions
Here are some sample questions you can test:
1. **Simple Arithmetic**:
   - `solve: There are 10 birds on a tree. 5 birds fly away. How many birds are left on the tree?`
   - `solve: There are 8 apples in a basket. 3 apples are taken out. How many apples are left in the basket?`

2. **Word Problems**:
   - `solve: Tom has 5 apples. He buys 3 more apples. How many apples does Tom have now?`
   - `solve: Lisa has 12 candies. She gives 4 candies to her friend. How many candies does Lisa have left?`

---

## Troubleshooting
### Common Errors
1. **Missing Dependencies**:
   - Install missing libraries using:
     ```bash
     pip install <library-name>
     ```

2. **Model Loading Issues**:
   - Ensure the model is downloaded and cached properly.
   - Clear Hugging Face cache if needed:
     ```bash
     rm -rf ~/.cache/huggingface
     ```

3. **Incorrect Prefix**:
   - Ensure the prefix matches the training data. Use `"solve: "` for inference.

---

## Development Notes
### Folder Structure
- `main.py`: Contains the GUI and inference logic.
- `t5_gsm8k_model/`: Folder for your fine-tuned T5-small model (ignored in `.gitignore`).
- `.gitignore`: Specifies files and folders to exclude from version control.

---

## License
This project is licensed under the MIT License.

---

## Author
**Shahzaib Khan**

---

## Acknowledgments
- Hugging Face Transformers
- GSM8K Dataset
- PyQt5 for GUI Development
