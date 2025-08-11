Offline Biblical Bot
A simple, offline Bible study assistant that runs entirely on your local machine—no API keys, accounts, or internet required.
This app uses a compact local language model combined with a Biblical knowledge base to answer your questions directly from Scripture.

The bot automatically builds a searchable vector store from the provided Bible text and uses a locally stored LLM to respond with relevant verses and summaries. Perfect for private, distraction-free study.


✨ Features
📖 Bible-Only Answers – All responses are drawn directly from the provided Biblical text.

🛠 Offline Capability – No external connections required after model download.

🔍 Semantic Search – Finds relevant verses, even with fuzzy queries.

💬 Natural Language Summaries – Understand Scripture passages quickly.

📦 Installation
Clone the repository

bash
Copy code
git clone https://github.com/YOUR_USERNAME/biblical-bot-terminal.git
cd biblical-bot-terminal
Install dependencies

bash
Copy code
pip install -r requirements.txt
Add your Bible text file
Place your JSON Bible data into the data/ folder.

Run the app

bash
Copy code
python main.py
🚀 Usage
When prompted, type your question:

perl
Copy code
What does the Bible say about forgiveness?
The bot will:

Search relevant verses in your Bible dataset

Summarize the answer in plain language

Provide exact verse citations

⚠️ Model Download
On first run, the bot will automatically download the appropriate small LLM file into models/.
Note: Large model files are not included in the repository to keep it under GitHub’s size limits.
