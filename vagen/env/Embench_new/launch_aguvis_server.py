import argparse
import time
from flask import Flask, request, jsonify

# Import your AguvisModel from wherever it's defined.
# from my_model_file import AguvisModel
# For this example, we'll assume it's in "my_model_module.py".
from embodiedbench.planner.aguvis_model import AguvisModel

app = Flask(__name__)

# We'll keep a global reference to the loaded model so it's accessible in the /respond endpoint.
model = None

@app.route("/respond", methods=["POST"])
def respond():
    """
    Expects a JSON payload of the form:
    {
      "prompt": "...",
      "obs": "...",
      "previous_actions": "...",
      "low_level_instruction": "...",
      "mode": "self-plan",
      "temperature": 0.0,
      "max_new_tokens": 1024
    }

    Returns:
    {
      "response": "Model-generated text"
    }
    """

    data = request.get_json(force=True) or {}

    messages = data.get("message", [])
    # prompt = data.get("prompt", "")
    # obs = data.get("obs", None)
    # previous_actions = data.get("previous_actions", None)
    # low_level_instruction = data.get("low_level_instruction", None)
    mode = data.get("mode", "self-plan")
    temperature = data.get("temperature", 0.1)
    max_new_tokens = data.get("max_new_tokens", 1024)

    try:
        response_text = model.respond(
            messages = messages,
            # prompt=prompt,
            # obs=obs,
            # previous_actions=previous_actions,
            # low_level_instruction=low_level_instruction,
            mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        # Optionally handle or log the error
        print("An unexpected error occurred:", e)
        # Retry once, similar to your original logic
        time.sleep(20)
        response_text = model.respond(
            messages = messages,
            # prompt=prompt,
            # obs=obs,
            # previous_actions=previous_actions,
            # low_level_instruction=low_level_instruction,
            mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

    return jsonify({"response": response_text})

def main():
    parser = argparse.ArgumentParser(description="Aguvis Model Server")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Embodied-reasoning-agent-uiuc/EB-Man-Qwen2.5-3b-instruct-sft-stage2", #~/era-checkpoints/EB-Man-Qwen2.5-3b-instruct-sft-stage2, Embodied-reasoning-agent-uiuc/Qwen2.5-VL-3B-Instruct-alfred-sft-full
        help="Path to the Qwen2.5VL (or any other) model."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:2",
        help="Device for running the model (e.g. 'cuda' or 'cpu')."
    )

    args = parser.parse_args()

    # Load the model once at server startup.
    global model
    model = AguvisModel(model_path=args.model_path, device=args.device)

    # Run the Flask app. Adjust host/port if needed.
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()