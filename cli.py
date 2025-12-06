#!/usr/bin/env python3
"""
Command-line interface for Research Assistant AI
Usage: python cli.py "your research query here"
"""

import sys
import argparse
from agent import research_query
from config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_separator():
    print("\n" + "=" * 80 + "\n")


def display_results(result):
    """Display research results in a readable format"""
    
    print_separator()
    print("📊 SYNTHESIS")
    print_separator()
    print(result.get('synthesis', 'No synthesis available'))
    
    print_separator()
    print("📚 TOP PAPERS")
    print_separator()
    
    papers = result.get('reranked_papers', [])
    if not papers:
        print("No papers found.")
        return
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.get('title', 'Unknown Title')}")
        
        authors = paper.get('authors', [])
        if authors:
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
            print(f"   Authors: {author_str}")
        
        year = paper.get('year', paper.get('published', 'N/A'))
        print(f"   Year: {year}")
        
        if paper.get('citation_count'):
            print(f"   Citations: {paper.get('citation_count')}")
        
        if paper.get('venue'):
            print(f"   Venue: {paper.get('venue')}")
        
        abstract = paper.get('abstract', '')
        if abstract:
            # Truncate long abstracts
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            print(f"   Abstract: {abstract}")
        
        # Links
        if paper.get('pdf_url'):
            print(f"   PDF: {paper.get('pdf_url')}")
        if paper.get('url'):
            print(f"   URL: {paper.get('url')}")
    
    print_separator()
    print("📈 STATISTICS")
    print_separator()
    print(f"Papers Found: {len(result.get('raw_papers', []))}")
    print(f"Retrieved: {len(result.get('retrieved_papers', []))}")
    print(f"Final Selection: {len(papers)}")
    print(f"Workflow Steps: {result.get('step_count', 0)}")
    
    if result.get('messages'):
        print_separator()
        print("🔍 WORKFLOW STEPS")
        print_separator()
        for msg in result.get('messages', []):
            print(f"  • {msg}")
    
    print_separator()


def main():
    parser = argparse.ArgumentParser(
        description='Research Assistant AI - CLI Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "retrieval augmented generation"
  python cli.py "transformer attention mechanisms" --context "I'm building a chatbot"
  python cli.py "few-shot learning" --verbose
        """
    )
    
    parser.add_argument(
        'query',
        type=str,
        help='Research query to search for'
    )
    
    parser.add_argument(
        '--context', '-c',
        type=str,
        default='',
        help='Additional context for the research query'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure your .env file is set up correctly with ANTHROPIC_API_KEY")
        sys.exit(1)
    
    # Display query info
    print_separator()
    print("🔬 RESEARCH ASSISTANT AI")
    print_separator()
    print(f"Query: {args.query}")
    if args.context:
        print(f"Context: {args.context}")
    print("\nSearching and analyzing papers... This may take a minute.")
    
    # Run the research query
    try:
        result = research_query(args.query, args.context)
        
        # Check for errors
        if result.get('error'):
            logger.error(f"Error during research: {result['error']}")
            sys.exit(1)
        
        # Display results
        display_results(result)
        
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
