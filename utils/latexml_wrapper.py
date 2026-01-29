"""
LaTeXML Wrapper for LaTeX → MathML Conversion

Installation:
  sudo apt-get install latexml  # Ubuntu/Debian
  # OR
  conda install -c conda-forge latexml  # Conda

Usage:
  from utils.latexml_wrapper import LaTeXMLConverter
  
  converter = LaTeXMLConverter()
  mathml = converter.convert(r"\frac{a}{b}")
"""

import subprocess
import tempfile
import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LaTeXMLConverter:
    """
    Python wrapper for LaTeXML command-line tool
    """
    
    def __init__(
        self,
        latexml_path: str = "latexml",
        latexmlmath_path: str = "latexmlmath",
        timeout: int = 10
    ):
        """
        Args:
            latexml_path: Path to latexml executable
            latexmlmath_path: Path to latexmlmath executable (for inline math)
            timeout: Max seconds per conversion
        """
        self.latexml_path = latexml_path
        self.latexmlmath_path = latexmlmath_path
        self.timeout = timeout
        
        # Check if LaTeXML is installed
        self.available = self._check_availability()
        
        if self.available:
            logger.info("✅ LaTeXML is available")
        else:
            logger.warning("⚠️ LaTeXML not found, conversion will be disabled")
    
    def _check_availability(self) -> bool:
        """Check if LaTeXML is installed"""
        try:
            result = subprocess.run(
                [self.latexmlmath_path, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert(self, latex: str) -> Optional[str]:
        """
        Convert LaTeX to MathML
        
        Args:
            latex: LaTeX formula string (e.g., r"\frac{a}{b}")
            
        Returns:
            MathML string, or None if conversion fails
        """
        if not self.available:
            logger.debug("LaTeXML not available, skipping conversion")
            return None
        
        if not latex:
            return None
        
        # Sanitize input
        latex = latex.strip()
        
        # Remove outer delimiters if present
        latex = re.sub(r'^\$+|\$+$', '', latex)
        latex = re.sub(r'^\\begin\{equation\*?\}|\\end\{equation\*?\}$', '', latex)
        
        try:
            # Use latexmlmath for inline formulas (faster than full latexml)
            result = subprocess.run(
                [self.latexmlmath_path, latex],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                mathml = result.stdout.strip()
                
                # Clean up MathML output
                mathml = self._clean_mathml(mathml)
                
                return mathml
            else:
                logger.warning(f"LaTeXML conversion failed: {result.stderr[:100]}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning(f"LaTeXML conversion timeout for: {latex[:50]}...")
            return None
        except Exception as e:
            logger.error(f"LaTeXML conversion error: {e}")
            return None
    
    def _clean_mathml(self, mathml: str) -> str:
        """Clean and extract core MathML content"""
        if not mathml:
            return ""
        
        # Remove XML declaration
        mathml = re.sub(r'<\?xml[^?]*\?>', '', mathml)
        
        # Remove unnecessary namespaces
        mathml = re.sub(r'\s*xmlns[^"]*="[^"]*"', '', mathml)
        
        # Extract <math> content
        match = re.search(r'<math[^>]*>(.*?)</math>', mathml, re.DOTALL)
        if match:
            mathml = f"<math>{match.group(1)}</math>"
        
        return mathml.strip()
    
    def extract_skeleton(self, mathml: str) -> str:
        """
        Extract MathML skeleton (same as in prepare scripts)
        
        Args:
            mathml: MathML XML string
            
        Returns:
            Comma-separated tag sequence
        """
        if not mathml:
            return ""
        
        try:
            # Remove attributes
            mathml = re.sub(r'\s+[a-z]+="[^"]*"', '', mathml)
            
            # Extract tag names
            tags = re.findall(r'<([a-zA-Z0-9]+)', mathml)
            
            # Filter out non-semantic tags
            ignored = {
                'math', 'semantics', 'annotation', 'annotation-xml',
                'mstyle', 'mrow', 'mtext', 'mspace'
            }
            
            semantic_tags = [t.lower() for t in tags if t.lower() not in ignored]
            
            return ",".join(semantic_tags) if semantic_tags else ""
        except Exception as e:
            logger.error(f"Skeleton extraction failed: {e}")
            return ""
    
    def convert_batch(self, latex_list: list, show_progress: bool = True):
        """
        Batch convert LaTeX formulas to MathML
        
        Args:
            latex_list: List of LaTeX strings
            show_progress: Show progress bar
            
        Returns:
            List of (latex, mathml, mathml_skel) tuples
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(latex_list, desc="Converting to MathML")
        else:
            iterator = latex_list
        
        for latex in iterator:
            mathml = self.convert(latex)
            mathml_skel = self.extract_skeleton(mathml) if mathml else ""
            
            results.append((latex, mathml, mathml_skel))
        
        return results


# ============================================================
# Batch Processing Script
# ============================================================
def enhance_queries_with_latexml(
    input_file: str = "data/processed/queries_final.json",
    output_file: str = "data/processed/queries_with_real_mathml.json"
):
    """
    Enhance queries by converting LaTeX to real MathML using LaTeXML
    
    This replaces pseudo-MathML with real MathML for better precision.
    """
    import json
    from pathlib import Path
    
    logger.info(f"📂 Loading queries from {input_file}")
    
    with open(input_file, 'r') as f:
        queries = json.load(f)
    
    converter = LaTeXMLConverter()
    
    if not converter.available:
        logger.error("❌ LaTeXML not available, cannot proceed")
        logger.error("   Install with: sudo apt-get install latexml")
        return
    
    # Convert queries
    updated = 0
    failed = 0
    
    logger.info(f"🔄 Converting {len(queries)} queries...")
    
    from tqdm import tqdm
    for qid, qdata in tqdm(queries.items(), desc="LaTeXML Conversion"):
        latex = qdata.get('latex', '')
        
        if not latex:
            continue
        
        # Only convert if using pseudo-MathML
        if qdata.get('mathml_source') == 'pseudo_mathml':
            mathml = converter.convert(latex)
            
            if mathml:
                mathml_skel = converter.extract_skeleton(mathml)
                
                qdata['mathml_raw'] = mathml
                qdata['mathml_skel'] = mathml_skel
                qdata['mathml_source'] = 'latexml_conversion'
                
                updated += 1
            else:
                failed += 1
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Conversion complete!")
    logger.info(f"   Updated: {updated}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Output: {output_path}")


if __name__ == "__main__":
    # Test conversion
    converter = LaTeXMLConverter()
    
    test_formulas = [
        r"\frac{a}{b}",
        r"x^2 + y^2 = z^2",
        r"\int_0^1 f(x) dx",
        r"\sqrt{\frac{a+b}{c}}"
    ]
    
    print("Testing LaTeXML Conversion:")
    print("=" * 60)
    
    for latex in test_formulas:
        mathml = converter.convert(latex)
        skel = converter.extract_skeleton(mathml) if mathml else None
        
        print(f"\nLaTeX: {latex}")
        print(f"MathML: {mathml[:100] if mathml else 'FAILED'}...")
        print(f"Skeleton: {skel}")
    
    print("\n" + "=" * 60)
    
    # Batch processing (uncomment to run)
    # enhance_queries_with_latexml()