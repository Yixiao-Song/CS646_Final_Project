import os
import pdb
import json
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

def prepare_context(
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

def extract_wiki_links(combined_results):
    wiki_links = []
    for query_data in combined_results.values():
        results = query_data["retrieve_results"]
        links = [pair[0] for pair in results]
        wiki_links.extend(links)
    return wiki_links

def prepare_context_rerank(dict_item, wiki_url_contents_dict, key_to_links="wiki_links", return_links=True):
    context = ""

    if key_to_links == "wiki_links":
        wiki_links = dict_item[key_to_links]
    elif "decomp" in key_to_links:
        wiki_links = extract_wiki_links(dict_item[key_to_links])
    else:
        wiki_links = [x[0] for x in dict_item[key_to_links]]
    
    for wiki_link in wiki_links:
        all_links.append(wiki_link)
        wiki_key_dict = wiki_url_contents_dict.get(wiki_link, "")
        if not wiki_key_dict: 
            print("No wiki_key_dict found for wiki_link:", wiki_link)
            pdb.set_trace()
        title = wiki_key_dict["title"]
        contents = wiki_key_dict["contents"]
        context += f"{n}.\n\n{title}\n\n{contents}\n\n"
    
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

rerank_prompt_template = """Given a query and a set of retrieved documents for this query, select the top five most relevant documents. Return the indices of the documents in the order of their relevance, separated by commas.

Query:
{question}

Retrieved Documents:
{context}

The indices of the five documents most relevant to the query are as follows, separated by commas:"""

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