#!/usr/bin/env python3
"""
Example usage scripts for Research Assistant AI
Demonstrates various ways to use the system
"""

from agent import research_query, ResearchAgent
from config import Config
import json


def example_1_simple_query():
    """Example 1: Simple research query"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Query")
    print("="*70)
    
    result = research_query("retrieval augmented generation")
    
    print("\nSynthesis:")
    print(result['synthesis'])
    
    print(f"\nFound {len(result['reranked_papers'])} relevant papers")


def example_2_contextual_query():
    """Example 2: Query with additional context"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Query with Context")
    print("="*70)
    
    result = research_query(
        query="few-shot learning",
        context="I'm working on a project that requires learning from very few examples in a low-resource domain."
    )
    
    print("\nSynthesis:")
    print(result['synthesis'][:500] + "...")
    
    print("\nTop 3 Papers:")
    for i, paper in enumerate(result['reranked_papers'][:3], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:2])}")
        print(f"   Year: {paper.get('year', paper.get('published', 'N/A'))}")


def example_3_detailed_workflow():
    """Example 3: Using agent with detailed access to workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Detailed Workflow Access")
    print("="*70)
    
    agent = ResearchAgent()
    result = agent.run(
        query="transformer attention mechanisms",
        user_context="Focus on efficiency improvements"
    )
    
    # Access workflow details
    print(f"\nWorkflow completed in {result['step_count']} steps:")
    for msg in result['messages']:
        print(f"  - {msg}")
    
    print(f"\nRetrieved {len(result['retrieved_papers'])} papers")
    print(f"Re-ranked to top {len(result['reranked_papers'])} papers")
    
    # Access specific papers
    if result['reranked_papers']:
        top_paper = result['reranked_papers'][0]
        print(f"\nTop Paper: {top_paper['title']}")
        print(f"Abstract: {top_paper['abstract'][:200]}...")


def example_4_comparing_papers():
    """Example 4: Comparing papers on a specific topic"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Papers")
    print("="*70)
    
    result = research_query(
        query="compare BERT and GPT architectures",
        context="I need to understand the key differences and use cases for each"
    )
    
    print("\nComparative Analysis:")
    print(result['synthesis'])


def example_5_finding_research_gaps():
    """Example 5: Identifying research gaps"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Finding Research Gaps")
    print("="*70)
    
    result = research_query(
        query="research gaps in multimodal learning",
        context="Looking for unexplored areas and future directions in multimodal AI"
    )
    
    print("\nIdentified Gaps and Directions:")
    print(result['synthesis'])


def example_6_domain_specific():
    """Example 6: Domain-specific research"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Domain-Specific Query")
    print("="*70)
    
    result = research_query(
        query="graph neural networks for drug discovery",
        context="Specifically interested in molecular property prediction and drug-target interaction"
    )
    
    print(f"\nFound {len(result['raw_papers'])} papers in total")
    print(f"Narrowed down to {len(result['reranked_papers'])} most relevant")
    
    print("\nKey Findings:")
    print(result['synthesis'][:400] + "...")


def example_7_export_results():
    """Example 7: Export results to JSON"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Export Results")
    print("="*70)
    
    result = research_query("neural machine translation")
    
    # Prepare data for export
    export_data = {
        'query': result['query'],
        'synthesis': result['synthesis'],
        'papers': [
            {
                'title': p['title'],
                'authors': p['authors'],
                'year': p.get('year', p.get('published')),
                'abstract': p['abstract'],
                'url': p.get('url') or p.get('pdf_url')
            }
            for p in result['reranked_papers']
        ],
        'metadata': {
            'total_found': len(result['raw_papers']),
            'final_count': len(result['reranked_papers']),
            'steps': result['step_count']
        }
    }
    
    # Save to file
    with open('research_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("\n✅ Results exported to research_results.json")
    print(f"Exported {len(export_data['papers'])} papers")


def example_8_iterative_research():
    """Example 8: Iterative research (multiple queries)"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Iterative Research")
    print("="*70)
    
    queries = [
        "What is retrieval augmented generation?",
        "How is RAG evaluated?",
        "What are the challenges in RAG systems?"
    ]
    
    agent = ResearchAgent()
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = agent.run(query)
        
        # Brief summary
        synthesis = result['synthesis']
        print(f"Summary: {synthesis[:200]}...")
        print(f"Papers: {len(result['reranked_papers'])}")


def run_all_examples():
    """Run all examples (comment out ones you don't want)"""
    
    print("\n")
    print("█" * 70)
    print("  RESEARCH ASSISTANT AI - EXAMPLE USAGE")
    print("█" * 70)
    
    try:
        # Validate config first
        Config.validate()
        
        # Run examples (comment out to skip)
        example_1_simple_query()
        # example_2_contextual_query()
        # example_3_detailed_workflow()
        # example_4_comparing_papers()
        # example_5_finding_research_gaps()
        # example_6_domain_specific()
        # example_7_export_results()
        # example_8_iterative_research()
        
        print("\n" + "="*70)
        print("✅ Examples completed successfully!")
        print("="*70)
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please ensure your .env file is set up with ANTHROPIC_API_KEY")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        example_func = f"example_{example_num}"
        
        if example_func in globals():
            print(f"\nRunning Example {example_num}...")
            globals()[example_func]()
        else:
            print(f"Example {example_num} not found")
            print("Available examples: 1-8")
    else:
        # Run all (or modify run_all_examples to select which ones)
        run_all_examples()
