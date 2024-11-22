from urllib.parse import unquote, quote

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
