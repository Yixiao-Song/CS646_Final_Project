import os
import pdb
import json
import re
from urllib.parse import unquote

def normalize_url(url):
    """Normalize URLs by decoding and re-encoding them

    Args:
        url: the unnormalized url from wikipedia dataset

    Return:
        normalized_url: readable url link
    """
    # Decode percent-encoded parts
    decoded_url = unquote(url)
    # Re-encode spaces as underscores (for Wikipedia consistency)
    normalized_url = decoded_url.replace(" ", "_")
    return normalized_url

def prepare_context_oracle(
        dict_item, wiki_url_contents_dict, key_to_links="wiki_links"
        ):
    """Prepare the context for the oracle prompting.
    For each wiki link in the grouth truth data, extract the title and
    contents from the wiki_url_contents_dict, and concatenate all of
    them to be the prompting context.

    Args:
        dict_item (dict): a dictionary item from the frames dataset.
            Important keys: "Prompt", "wiki_links"
        wiki_url_contents_dict (dict): a dictionary of wiki urls to a
        dictionary containing id, title, and contents.
    
    Returns:
        context (str): the concatenated context for the oracle prompting.
    """
    # initialize context for oracle prompting
    context = ""

    if key_to_links == "wiki_links":
        wiki_links = dict_item[key_to_links]
    else:
        wiki_links = [x[0] for x in dict_item[key_to_links]]

    for wiki_link in wiki_links:
        wiki_key_dict = wiki_url_contents_dict.get(wiki_link, "")
        if not wiki_key_dict: 
            print("No wiki_key_dict found for wiki_link:", wiki_link)
            pdb.set_trace()

        title = wiki_key_dict["title"]
        contents = wiki_key_dict["contents"]
        
        context += f"{title}\n\n{contents}\n\n"
    
    return context

def extract_wiki_links(combined_results, topk=None):
    wiki_links = []
    for query_data in combined_results.values():
        results = query_data["retrieve_results"]
        links = [pair[0] for pair in results]
        if topk:
            links = links[:topk]
        wiki_links.extend(links)
    return wiki_links

def prepare_context(dict_item, wiki_url_contents_dict, key_to_links="wiki_links", return_links=True, topk=None):
    context = ""
    if "rerank" in key_to_links:
        wiki_links = extract_wiki_links(dict_item[key_to_links], topk=topk)
    else:
        wiki_links = [x[0] for x in dict_item[key_to_links]]
    
    for i, wiki_link in enumerate(wiki_links):
        wiki_key_dict = wiki_url_contents_dict.get(wiki_link, "")
        if not wiki_key_dict: 
            print("No wiki_key_dict found for wiki_link:", wiki_link)
            pdb.set_trace()
        title = wiki_key_dict["title"]
        contents = wiki_key_dict["contents"]
        context += f"{i+1}.\nTitle: {title}\nContent: {contents}\n\n"
    context = context.strip()

    if return_links:
        return context, wiki_links
    return context

def is_integer_list(text):
    if not text:
        return False
    items = [item.strip() for item in text.split(',')]
    if any(not item for item in items):
        return False
    try:
        [int(item) for item in items]
        return True
    except ValueError:
        return False

def find_integer_list(text, n=5):
    pattern = r'(-?\d+\s*,\s*){' + str(n-1) + r'}-?\d+'
    matches = re.finditer(pattern, text)
    for match in matches:
        number_string = match.group(0)
        try:
            numbers = [int(num.strip()) for num in number_string.split(',')]
            if len(numbers) == n:
                return True, numbers
        except ValueError:
            continue
    return False, None

def get_start_point(file_path):
    """Get the starting point for appending to a file.
    This function checks if the generation has started before and find the pick-up point.

    Args:
        file_path (str): the output file path to check for the starting
        point.
    
    Returns:
        start_point (int): the starting point for appending to the file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            start_point = len([json.loads(x.strip()) \
                            for x in f.readlines() if x.strip()])
    else:
        start_point = 0

    return start_point

oracle_prompt_template = """Given the context, answer the question with concise answers.

Context:
{context}

Question:
{question}

Given the context, the answer to the question:"""

auto_eval_prompt_template = """Your task is to decide whether Answer 1 is entailed in Answer 2. If it is entailed, return "yes", otherwise return "no". Do not return anything else.

Examples:
Question: If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? 
Answer 1: Jane Ballou
Answer 2: To solve this:

15th First Lady's Mother (First Name):

The 15th First Lady of the United States was Harriet Lane, the niece of President James Buchanan. Harriet Lane's mother was Jane Lane. Thus, the first name of the 15th First Lady's mother is Jane.
Second Assassinated President's Mother's Maiden Name (Surname):

The second assassinated president was James A. Garfield. His mother was Eliza Ballou Garfield, and her maiden name was Ballou.

Your future wife's name is Jane Ballou.
Judgement: yes

Question: Imagine there is a building called Bronte tower whose height in feet is the same number as the dewey decimal classification for the Charlotte Bronte book that was published in 1847. Where would this building rank among tallest buildings in New York City, as of August 2024?
Answer 1: 37th
Answer 2: The Dewey Decimal Classification (DDC) number for Charlotte Brontë's 1847 novel, Jane Eyre, is 823.8. 
OPEN LIBRARY
 If Brontë Tower were constructed with a height of 823.8 feet, it would rank among the tallest buildings in New York City.

As of August 2024, the tallest buildings in New York City are:

One World Trade Center: 1,776 feet
Central Park Tower: 1,550 feet
111 West 57th Street (Steinway Tower): 1,428 feet
One Vanderbilt: 1,401 feet
432 Park Avenue: 1,396 feet
WIKIPEDIA

With a height of 823.8 feet, Brontë Tower would be taller than several notable buildings, such as:

The New York Times Building: 1,046 feet
Chrysler Building: 1,046 feet
Bank of America Tower: 1,200 feet
Therefore, Brontë Tower would rank approximately 20th among the tallest buildings in New York City as of August 2024.
Judgement: no

Your turn: If Answer 1 is entailed in Answer 2, return "yes", otherwise return "no". Do not return anything else.

Question: {question}
Answer 1: {ground_truth_answer}
Answer 2: {predicted_answer}
Judgement: """

extract_query_prompt_template = """Please break down the following query into a list of {max_num} subqueries. The extracted subqueries must satisfy the following requirements:

1. Self-containment: Each subquery should be complete and self-contained, as if it were being asked in isolation. It should not require knowledge of the original query or other subqueries.
2. Full coverage: The subqueries, when combined, should fully address the scope of the original query.
3. Equal specificity: Each subquery should be equally detailed, avoiding any being overly broad or overly narrow compared to the others.

Your output should be in JSON format, where the key is 'atomic_questions' and the value is the list of your extracted subqueries.

Query: {question}
Extracted subqueries:"""

extract_query_dynamic_prompt_template = """Please break down the following query into a list of subqueries. The extracted subqueries must satisfy the following requirements:

1. Self-containment: Each subquery should be complete and self-contained, as if it were being asked in isolation. It should not require knowledge of the original query or other subqueries.
2. Full coverage: The subqueries, when combined, should fully address the scope of the original query.
3. Equal specificity: Each subquery should be equally detailed, avoiding any being overly broad or overly narrow compared to the others.

Guidance for Subquery Count: You are responsible for deciding how many subqueries to extract. This decision should be guided by the goal of fully satisfying the criteria above.
Output Format: Your output must be in JSON format. Use the key "atomic_questions" for the list of extracted subqueries.

Query: {question}
Extracted subqueries:"""

extract_query_dynamic_fewshot_prompt_template = """Please break down the following query into a list of subqueries. The extracted subqueries must satisfy the following requirements:

1. Self-containment: Each subquery should be complete and self-contained, as if it were being asked in isolation. It should not require knowledge of the original query or other subqueries.
2. Full coverage: The subqueries, when combined, should fully address the scope of the original query.
3. Equal specificity: Each subquery should be equally detailed, avoiding any being overly broad or overly narrow compared to the others.

The goal is to create subqueries that effectively contribute to answering the original query. Your output must be in JSON format. Use the key "atomic_questions" for the list of extracted subqueries.

Below are some examples:

Example query: Which movie musical produced a song that was inspired by poetry from an American poet, who was born a week after Queen Victoria?
Example extracted subqueries:
{{
  "atomic_questions": [
    "When was Queen Victoria born?",
    "Which American poet was born a week after Queen Victoria?",
    "Which songs from movie musicals were inspired by American poetry?"
  ]
}}

Example query: Who was the Catholic Pope eleven years after Emperor Charlemagne died?
Example extracted subqueries:
{{
  "atomic_questions": [
    "When did Emperor Charlemagne die?",
    "Who have served as Catholic Popes throughout history?"
  ]
}}

Example query: What is greater: the combined 2011 populations of Rennington (Northumberland), Lydbrook (Gloucestershire), Stow-on-the-Wold (Gloucestershire) and Witney (Oxfordshire), or the 2022 population of London?
Example extracted subqueries:
{{
  "atomic_questions": [
    "What was the population of Rennington (Northumberland) in 2011?",
    "What was the population of Lydbrook (Gloucestershire) in 2011?",
    "What was the population of Stow-on-the-Wold (Gloucestershire) in 2011?",
    "What was the population of Witney (Oxfordshire) in 2011?",
    "What was the population of London in 2022?"
  ]
}}

Your query: {question}
Your extracted subqueries:"""
