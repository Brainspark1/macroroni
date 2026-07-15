import os
import subprocess
import logging

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments

logger = logging.getLogger("nes_voice.trainer")

class NESBERTTrainer:
    def __init__(self, device_backend="mps"):
        """
        :param base_model: The starting core weights to fine-tune (defaults to your universal model).
        :param device_backend: hardware backend currently found in user's computer (write either "mps" for MacOS, "cuda" for Nvidia, or "cpu")
        """

        self.base_model = "Saggarwal/token_bert"
        self.device_backend = device_backend
        
    # method to invoke chatette to compile template file passed into raw training sentences and intents labels
    def compile_chatette_template(self, chatette_path, output_dir="./generated_dataset"):
        logger.info(f"Compiling Chatette template file at {chatette_path}")

        if not os.path.exists(chatette_path):
            raise FileNotFoundError(f"Chatette template file not found at {chatette_path}")

        # create directory to store the generated dataset to be accessed easier by the bert model later on  
        os.makedirs(output_dir, exist_ok=True)
        
        # executing chatette cli using subprocess to generate dataset
        try:
            subprocess.run([
                "python", "-m", "chatette", 
                chatette_path, 
                "-o", output_dir
            ], check=True)

            logger.info(f"Chatette dataset successfully generated inside: {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Chatette compilation failed: {e}")
            raise e

    # placeholder method to parse chatette output, tokenize words, returns huggingface dataset object - not implemented yet
    def _prepare_hf_dataset(self, generated_dir, tokenizer):

        pass

    # method to compile chatette template, process data, fine-tune core bert model
    def train(
        self, 
        chatette_path, 
        output_model_dir="./trained_game_model", 
        epochs=3, 
        batch_size=8
    ):
        """
        :param chatette_path: path to customized .chatette dataset file
        :param output_model_dir: local path where your final game-specific weights of the customized bert model will save
        """

        # generating text from chatette file into dataset
        generated_dir = "./tmp_chatette_out"
        self.compile_chatette_template(chatette_path, output_dir=generated_dir)
        
        # setting up tokenizer and model
        logger.info(f"Loading base model core {self.base_model}")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForTokenClassification.from_pretrained(self.base_model)
        
        # building dataset - commented out as depends on prepare huggingface dataset method not implemented above
        # tokenized_dataset = self._prepare_hf_dataset(generated_dir, tokenizer)
        
        # configuring hardware errors - note that huggingface trainer handles cuda automatically if present
        use_mps = True if self.device_backend == "mps" else False

        training_args = TrainingArguments(
            output_dir="./nesbert_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_steps=10,
            use_mps_device=use_mps,
            report_to="none" # getting rid of external trackers to prevent unwanted setups or polluting workspace
        )
        
        logger.info("Beginning to fine-tune NESBERT")
        
        # section commented out as relies on tokenized_dataset that relies on the prepare huggingfacce dataset method that is not yet implemented
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=tokenized_dataset,
        #     data_collator=DataCollatorForTokenClassification(tokenizer)
        # )
        # trainer.train()
        
        # saving new weights to local location
        logger.info(f"Saving fine-tuned game model to {output_model_dir}")

        model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)

        logger.info("Training complete")