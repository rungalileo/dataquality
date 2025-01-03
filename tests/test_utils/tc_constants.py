from datasets import Dataset

TC_JSON_DATA = [
    {"text": "Card not working, help?", "label": 4},
    {"text": "Card declined, why?", "label": 4},
    {"text": "Can't use card, what's wrong?", "label": 4},
    {"text": "My card isn't working at the ATM, what should I do?", "label": 4},
    {"text": "I'm having trouble using my card for online payments", "label": 4},
    {"text": "My card isn't working on Amazon, can you help", "label": 4},
    {"text": "Card keeps getting declined at Starbucks, what should I do", "label": 4},
    {"text": "Unable to use my card for Uber, is there an issue", "label": 4},
    {"text": "Fraudulent charge, help?", "label": 11},
    {"text": "Unrecognized transaction, report?", "label": 11},
    {"text": "Unauthorized purchase, assist?", "label": 11},
    {"text": "Unauthorized transaction from eBay, need assistance", "label": 11},
    {"text": "Suspicious charge from Netflix, how to report", "label": 11},
    {"text": "Someone used my card on Walmart, what should I do", "label": 11},
    {"text": "Need account number, where?", "label": 0},
    {"text": "Find card number?", "label": 0},
    {"text": "Checking account number?", "label": 0},
    {"text": "How can I find my account number for my checking account?", "label": 0},
    {"text": "Can you provide my savings account number?", "label": 0},
    {"text": "Need my account number for online shopping", "label": 0},
    {"text": "Where can I find my card number for Amazon", "label": 0},
    {"text": "Provide my checking account number for Zelle", "label": 0},
    {"text": "Card not here yet?", "label": 6},
    {"text": "New card status?", "label": 6},
    {"text": "Card delivery update?", "label": 6},
    {"text": "New card hasn't arrived, please check status", "label": 6},
    {"text": "Waiting for my card, can you provide an update", "label": 6},
    {"text": "Requested card weeks ago, still not here", "label": 6},
    {"text": "Lost card, replace?", "label": 5},
    {"text": "New card, how?", "label": 5},
    {"text": "Stolen card, help?", "label": 5},
    {"text": "Lost my card, need a replacement for online purchases", "label": 5},
    {"text": "Debit card missing, how to get a new one", "label": 5},
    {"text": "Card stolen, need help ordering a new one", "label": 5},
    {"text": "Stop payment, how?", "label": 3},
    {"text": "Cancel transaction?", "label": 3},
    {"text": "Undo money transfer?", "label": 3},
    {"text": "Accidentally sent money via Zelle, can I cancel", "label": 3},
    {"text": "Stop a pending payment on Amazon, what's the process", "label": 3},
    {"text": "Need to cancel a scheduled Quickpay transfer, how", "label": 3},
    {"text": "Recent transactions?", "label": 12},
    {"text": "View account history?", "label": 12},
    {"text": "Check past payments?", "label": 12},
    {"text": "Can you show me my recent transactions on my account?", "label": 12},
    {"text": "How do I view my past payments and transfers?", "label": 12},
    {"text": "Where can I see a history of my account activity?", "label": 12},
    {"text": "Show recent transactions, including Amazon purchases", "label": 12},
    {"text": "View past payments sent via Zelle", "label": 12},
    {"text": "Check account activity, including Quickpay transfers", "label": 12},
    {"text": "Cancel purchase?", "label": 2},
    {"text": "Stop transaction?", "label": 2},
    {"text": "Undo charge?", "label": 2},
    {"text": "I need to cancel a recent purchase I made, can you help?", "label": 2},
    {"text": "How do I stop a pending transaction on my account?", "label": 2},
    {"text": "Can I cancel a subscription charge that's on my account?", "label": 2},
    {"text": "Cancel recent purchase on eBay, can you help", "label": 2},
    {"text": "Stop pending transaction from Apple Store", "label": 2},
    {"text": "Cancel a subscription charge from Spotify", "label": 2},
    {"text": "Card arrival date?", "label": 10},
    {"text": "New card ETA?", "label": 10},
    {"text": "Card delivery status?", "label": 10},
    {"text": "Can you give me an update on when my new card will arrive?", "label": 10},
    {"text": "What's the estimated delivery date for my new debit card?", "label": 10},
    {"text": "Update on new card arrival, needed for online shopping", "label": 10},
    {"text": "How long for a replacement card to be delivered", "label": 10},
    {"text": "Estimated delivery date for my new credit card", "label": 10},
    {"text": "Link new account?", "label": 1},
    {"text": "Add credit card?", "label": 1},
    {"text": "Connect new account?", "label": 1},
    {"text": "Add a new credit card for Amazon purchases", "label": 1},
    {"text": "Link a new account for Zelle transfers", "label": 1},
    {"text": "Connect a new account to my mobile banking app", "label": 1},
    {"text": "Bank is useless!", "label": 7},
    {"text": "Terrible service!", "label": 7},
    {"text": "Awful support!", "label": 7},
    {"text": "This bank is useless, you never help with my issues!", "label": 7},
    {"text": "This bank never helps with Amazon issues", "label": 7},
    {"text": "Frustrated with your service, can't use Zelle", "label": 7},
    {"text": "Support is a joke, Quickpay never works", "label": 7},
    {"text": "Interest fee, why?", "label": 9},
    {"text": "Unexplained charge?", "label": 9},
    {"text": "Wrong interest rate?", "label": 9},
    {"text": "Why was I charged interest on my account last month?", "label": 9},
    {"text": "There's an unexpected interest fee, can you explain?", "label": 9},
    {"text": "My interest rate seems incorrect, can you check it?", "label": 9},
    {"text": "Why was I charged interest after shopping on Amazon", "label": 9},
    {"text": "Unexpected interest fee on my account, explain", "label": 9},
    {"text": "Interest rate seems incorrect, please verify", "label": 9},
    {"text": "How do I set up online banking?", "label": 8},
    {"text": "Can I deposit checks through the mobile app?", "label": 8},
    {"text": "What's the daily ATM withdrawal limit?", "label": 8},
    {"text": "How do I reset my online banking password?", "label": 8},
    {"text": "How long does it take for a check to clear?", "label": 8},
    {"text": "What are the bank's hours of operation?", "label": 8},
    {"text": "Can I transfer money to another bank?", "label": 8},
    {"text": "How do I set up direct deposit?", "label": 8},
    {"text": "What are the fees for maintaining an account?", "label": 8},
    {"text": "How do I order checks?", "label": 8},
    {"text": "Are there any foreign transaction fees?", "label": 8},
    {"text": "Can I open a joint account?", "label": 8},
    {"text": "How do I find my routing number?", "label": 8},
    {"text": "What's the minimum balance required for my account?", "label": 8},
    {"text": "What should I do if I can't find my credit card?", "label": 8},
    {"text": "How do I report a lost or stolen debit card?", "label": 8},
    {"text": "What are the steps to take after misplacing my card?", "label": 8},
    {"text": "How do I report a lost card for my Amazon account", "label": 8},
    {"text": "Steps to take if my card is stolen and used on eBay", "label": 8},
    {"text": "What to do if I can't find my card for online shopping", "label": 8},
]

TC_LABELS = [
    "Account Number",
    "Add Account",
    "Cancel",
    "Cancel Payment",
    "Card Issue",
    "Card Lost_and_Card Replacement",
    "Card Not Received",
    "Customer Insult",
    "FAQ",
    "Interest charge",
    "Mailing status",
    "Report Fraud",
    "See Activity",
]

TRAIN_HF_DS = Dataset.from_list(TC_JSON_DATA)
# 🔭🌕 Galileo preprocessing
TRAIN_HF_DS = TRAIN_HF_DS.map(lambda x, idx: {"id": idx}, with_indices=True)

# Test dataset (no label names)
TEST_HF_DS = Dataset.from_list(TC_JSON_DATA)
TEST_HF_DS = TEST_HF_DS.map(lambda x, idx: {"id": idx}, with_indices=True)
