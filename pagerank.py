import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Number of pages in the corpus
    N = len(corpus)

    # Probability distribution initialization
    prob_dist = {pg: 0 for pg in corpus}

    # Links from the current page
    links = corpus[page]

    # If the current page has no outgoing links,
    # treat all pages as having equal probability
    if len(links) == 0:
        return {pg: 1/N for pg in corpus}

    # Calculate probability for each page
    for pg in corpus:
        # Base probability for each page due to the random jump
        prob_dist[pg] = (1 - damping_factor) / N

        # Additional probability if the current page links to this page
        if pg in links:
            prob_dist[pg] += damping_factor / len(links)

    for pg in corpus:
        prob_dist[pg] = round(prob_dist[pg], 3)

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(
            list(distribution.keys()),
            weights=distribution.values(),
            k=1
        )[0]
        pageranks[page] += 1

    pageranks = {page: rank / n for page, rank in pageranks.items()}
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    THRESHOLD = 0.001

    pageranks = {page: 1 / len(corpus) for page in corpus}

    converged = False
    while not converged:
        new_pageranks = {
            page: calculate_page_rank(
                page,
                damping_factor,
                corpus,
                pageranks
            )
            for page in corpus
        }

        converged = all(
            abs(new_pageranks[page] - pageranks[page]) < THRESHOLD
            for page in pageranks
        )
        # Update pageranks with new value
        pageranks = new_pageranks

    # Normalize the PageRank values
    total_rank = sum(pageranks.values())
    pageranks = {page: rank / total_rank for page, rank in pageranks.items()}

    return pageranks


def calculate_page_rank(page, damping_factor, corpus, pageranks):
    return (
        (1 - damping_factor) / len(corpus)
        +
        damping_factor * sum([
            pageranks[pg] / len(corpus[pg])
            for pg in corpus if page in corpus[pg]
        ])
    )


if __name__ == "__main__":
    main()
