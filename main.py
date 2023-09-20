import openai
import tiktoken
import imaplib
import email
from email.header import decode_header
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tenacity import retry, stop_after_attempt, wait_random_exponential
import datetime
import time

# Set up OpenAI API key
openai.api_key = "your open ai key"

# OpenAI system promptA
system_prompt = """
You are a relevant information extractor bot who finds ALL the relevant information from a given piece of bank email, 
and then extract the information in the sequence as defined as DATE, MODE OF PAYMENT, BANK, MERCHANT, CURRENCY, AMOUNT, CARD NUMBER, ACCOUNT NUMBER. There will be only one line of the content. SEPERATE THE HEADER AND DATA WITH COLON. just give the HORIZONTAL table, that's it.
"""

# Function to calculate the number of tokens in text
def token_calculator(text):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(list(encoder.encode(text)))

# Function to extract information using OpenAI's GPT-3
@retry(wait=wait_random_exponential(min=1, max=60), stop = stop_after_attempt(6))
def extract_information(order_confirmation_email):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": order_confirmation_email}
        ],
        temperature=0.01,
        max_tokens=150,
        stop=None
    )

    time.sleep(10)
    return response.choices[0].message["content"]

# Your existing code for SVM model loading and email preprocessing
# ...
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Your Gmail credentials
username = "your-username"
app_password = "your-password"   # Generate this from your Google Account settings

# Connect to Gmail's IMAP server
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(username, app_password)

# Select the mailbox you want to read emails from (e.g., "inbox")
mailbox = "inbox"
mail.select(mailbox)

# Define a regular expression pattern to identify URLs
url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# Function to preprocess email text (removing URLs, HTML tags, empty lines)
def preprocess_text(text):
    text = re.sub(url_pattern, "", text)
    text = text.strip()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

# Function to preprocess email text (removing punctuation marks, stop words)
def preprocess_text_advanced(text):
    text = text.lower()
    modified_punctuation = string.punctuation.replace('.', '').replace(',', '')
    text = text.translate(str.maketrans('', '', modified_punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=30)  # Change the number of days as needed

# Convert the start and end dates to the required format for the IMAP search query
start_date_str = start_date.strftime("%d-%b-%Y")
end_date_str = end_date.strftime("%d-%b-%Y")

# Use the date criteria in the IMAP search query to filter emails within the specified date range
search_criteria = f'(SINCE {start_date_str} BEFORE {end_date_str})'
# Search for all emails in the mailbox
status, email_ids = mail.search(None, search_criteria)
email_ids = email_ids[0].split()
# Iterate through email IDs and fetch email details
for idx, email_id in enumerate(email_ids, start=1):
    status, email_data = mail.fetch(email_id, "(RFC822)")
    raw_email = email_data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding if encoding else "utf-8")

    from_, encoding = decode_header(msg.get("From"))[0]
    if isinstance(from_, bytes):
        from_ = from_.decode(encoding if encoding else "utf-8")

    # If the email is in plain text or multipart format
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if "text/plain" in content_type:
                body = part.get_payload(decode=True)
                if isinstance(body, bytes):
                    try:
                        # Attempt to decode using utf-8, ignoring errors
                        body_text = body.decode("utf-8", "ignore")
                    except UnicodeDecodeError:
                        # If utf-8 decoding fails, try using iso-8859-1 (a more permissive encoding)
                        body_text = body.decode("iso-8859-1", "ignore")

                    # Preprocess the extracted email text (remove URLs, HTML tags, empty lines)
                    preprocessed_text = re.sub(r'http\S+', '', body_text, flags=re.MULTILINE)  # Remove URLs
                    preprocessed_text = preprocessed_text.strip()
                    preprocessed_text = "\n".join([line.strip() for line in preprocessed_text.splitlines() if line.strip()])

                    # Preprocess the extracted email text (remove punctuation marks, stop words)
                    preprocessed_text = preprocessed_text.lower()
                    #preprocessed_text = preprocessed_text.translate(str.maketrans('', '', string.punctuation))
                    words = word_tokenize(preprocessed_text)
                    stop_words = set(stopwords.words("english"))
                    words = [word for word in words if word not in stop_words]
                    preprocessed_text = " ".join(words)

                    # Vectorize the preprocessed text and make predictions using the SVM model
                    tfidf_matrix = vectorizer.transform([preprocessed_text])
                    predicted_label = svm_model.predict(tfidf_matrix)

                    if predicted_label == 1:
                        try:
                            # Print email information
                            print("Subject:", subject)
                            print("From:", from_)
                            #print("Content:", preprocessed_text)

                            # Extract information using OpenAI's GPT-3
                            extracted_info = extract_information(preprocessed_text)
                            print("\n",extracted_info)

                            #print(extracted_info)

                            #print("Predicted Label: Transaction Successful", end="\n\n")
                        except UnicodeEncodeError:
                            # Handle Unicode character errors if needed
                            pass

# Logout and close the connection
mail.logout()
