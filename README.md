# Lottery Simulation Web App (Gradio-Based)

This project is a web-based lottery simulation tool built using Python and Gradio. It is designed to simulate a prize-draw mechanism (commonly seen in systems such as Japanese Ichiban Kuji-style lotteries). Users can configure inventory, pricing, exchange rules, stop conditions, and other parameters to observe how the simulation behaves under different strategies.

---

## Features

- Configurable inventory with customizable colors, quantities, and pricing.
- Adjustable draw price per attempt.
- Support for exchange rules (e.g., convert lower-tier items into higher-tier ones).
- Multiple stopping conditions:
  - Stop when any target is reached
  - Stop when all targets are met
  - Option to stop immediately when a condition is fulfilled
- Support for multiple runs, number of draws per round, and seed control.
- Optional automatic random seed generation.
- Simulation output can be exported as a text report.
- User settings can be exported as JSON for reuse.

---

## Live Demo

You can access the deployed version here:

https://random-test-o81w.onrender.com

Note: The first load may take a few seconds if the server is waking up.

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python     | Core programming language |
| Gradio     | Web interface framework |
| Pandas     | Data handling and validation |
| Render     | Cloud hosting and deployment |

---

## Local Development

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application:
```bash
python simulator_gradio.py
```

The application will be available at:
```
http://127.0.0.1:7860
```

---

## Deployment Notes

When deploying to cloud platforms such as Render or Railway, ensure that the application listens to the environment-defined port. The following example is recommended in your main file:
```python
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
```

## License and Intended Use

This project is intended for educational and simulation purposes. For commercial use or extended customization, please contact the project owner.