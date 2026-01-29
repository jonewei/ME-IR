"""
完整的数据预处理工作流自动化脚本
一键解决所有已知问题并生成诊断报告
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🚀 核心1: 步骤执行器
# ============================================================
class WorkflowStep:
    def __init__(self, name, script, required_inputs=None, outputs=None):
        self.name = name
        self.script = script
        self.required_inputs = required_inputs or []
        self.outputs = outputs or []
        self.success = False
        self.error_msg = None
    
    def check_prerequisites(self):
        """检查前置文件是否存在"""
        missing = [f for f in self.required_inputs if not Path(f).exists()]
        if missing:
            return False, f"Missing files: {missing}"
        return True, None
    
    def execute(self):
        """执行步骤"""
        logger.info("="*60)
        logger.info(f"🚀 Executing: {self.name}")
        logger.info(f"   Script: {self.script}")
        logger.info("="*60)
        
        # 检查前置条件
        ok, msg = self.check_prerequisites()
        if not ok:
            logger.error(f"❌ Prerequisites not met: {msg}")
            self.error_msg = msg
            return False
        
        try:
            # 执行脚本
            result = subprocess.run(
                [sys.executable, self.script],
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 检查执行结果
            if result.returncode == 0:
                logger.info(f"✅ {self.name} completed successfully")
                
                # 验证输出文件
                missing_outputs = [f for f in self.outputs if not Path(f).exists()]
                if missing_outputs:
                    logger.warning(f"⚠️ Expected outputs not found: {missing_outputs}")
                    self.error_msg = f"Missing outputs: {missing_outputs}"
                    return False
                
                self.success = True
                return True
            else:
                logger.error(f"❌ {self.name} failed with code {result.returncode}")
                logger.error(f"   stderr: {result.stderr[:500]}")
                self.error_msg = result.stderr
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {self.name} timed out after 10 minutes")
            self.error_msg = "Execution timeout"
            return False
        
        except Exception as e:
            logger.error(f"❌ {self.name} failed with exception: {e}")
            self.error_msg = str(e)
            return False

# ============================================================
# 🚀 核心2: 数据验证器
# ============================================================
def validate_data_alignment():
    """
    验证数据对齐的正确性(关键诊断)
    """
    logger.info("="*60)
    logger.info("🔍 Validating data alignment...")
    logger.info("="*60)
    
    data_dir = Path("data/processed")
    
    # 加载文件
    files = {
        'queries': data_dir / "queries_final.json",
        'formulas': data_dir / "formulas.json",
        'relevance': data_dir / "relevance_labels.json"
    }
    
    data = {}
    for name, path in files.items():
        if not path.exists():
            logger.error(f"❌ {name} file not found: {path}")
            return False
        
        with open(path, 'r', encoding='utf-8') as f:
            data[name] = json.load(f)
    
    queries = data['queries']
    formulas = data['formulas']
    relevance = data['relevance']
    
    # 验证1: 查询与qrel的对齐
    logger.info("📊 Check 1: Query-Qrel Alignment")
    query_ids_in_qrel = set(relevance.keys())
    query_ids_in_file = set(queries.keys())
    
    common_queries = query_ids_in_qrel & query_ids_in_file
    logger.info(f"  Queries in qrel: {len(query_ids_in_qrel)}")
    logger.info(f"  Queries in file: {len(query_ids_in_file)}")
    logger.info(f"  Common: {len(common_queries)} ({len(common_queries)/len(query_ids_in_qrel)*100:.1f}%)")
    
    if len(common_queries) == 0:
        logger.error("❌ CRITICAL: No overlap between queries and qrel!")
        logger.error(f"   Sample qrel IDs: {list(query_ids_in_qrel)[:5]}")
        logger.error(f"   Sample query IDs: {list(query_ids_in_file)[:5]}")
        return False
    
    # 验证2: qrel中的doc_id是否在corpus中
    logger.info("📊 Check 2: Qrel-Corpus Alignment")
    all_relevant_docs = set()
    for query_rels in relevance.values():
        all_relevant_docs.update(query_rels.keys())
    
    docs_in_corpus = all_relevant_docs & set(formulas.keys())
    
    logger.info(f"  Relevant docs in qrel: {len(all_relevant_docs)}")
    logger.info(f"  Found in corpus: {len(docs_in_corpus)} ({len(docs_in_corpus)/len(all_relevant_docs)*100:.1f}%)")
    
    if len(docs_in_corpus) < len(all_relevant_docs) * 0.5:
        logger.error("❌ CRITICAL: Less than 50% of relevant docs found in corpus!")
        
        missing_sample = list(all_relevant_docs - docs_in_corpus)[:5]
        corpus_sample = list(formulas.keys())[:5]
        
        logger.error(f"   Sample missing doc IDs: {missing_sample}")
        logger.error(f"   Sample corpus IDs: {corpus_sample}")
        logger.error("   🔧 FIX: Increase corpus_shards in prepare_final_arqmath.py")
        return False
    
    # 验证3: 查询的MathML覆盖率
    logger.info("📊 Check 3: Query MathML Coverage")
    queries_with_mathml = sum(1 for q in queries.values() if q.get('mathml_skel'))
    
    logger.info(f"  Total queries: {len(queries)}")
    logger.info(f"  With MathML: {queries_with_mathml} ({queries_with_mathml/len(queries)*100:.1f}%)")
    
    if queries_with_mathml < len(queries) * 0.8:
        logger.warning("⚠️ WARNING: Less than 80% queries have MathML")
        logger.warning("   This may impact retrieval performance")
    
    # 验证4: LaTeX与MathML的一致性(采样检查)
    logger.info("📊 Check 4: LaTeX-MathML Consistency (Sample)")
    sample_size = min(10, len(queries))
    sample_queries = list(queries.values())[:sample_size]
    
    consistent = 0
    for qdata in sample_queries:
        has_latex = bool(qdata.get('latex'))
        has_mathml = bool(qdata.get('mathml_skel'))
        
        if has_latex and has_mathml:
            consistent += 1
    
    logger.info(f"  Sample size: {sample_size}")
    logger.info(f"  With both LaTeX & MathML: {consistent}/{sample_size}")
    
    # 验证5: ID格式一致性
    logger.info("📊 Check 5: ID Format Consistency")
    
    qrel_id_sample = list(all_relevant_docs)[:3]
    corpus_id_sample = list(formulas.keys())[:3]
    
    logger.info(f"  Sample qrel doc IDs: {qrel_id_sample}")
    logger.info(f"  Sample corpus IDs: {corpus_id_sample}")
    
    # 检查是否有格式冲突(如纯数字 vs 带前缀)
    qrel_numeric = all(doc_id.isdigit() for doc_id in qrel_id_sample if doc_id)
    corpus_numeric = all(fid.isdigit() for fid in corpus_id_sample if fid)
    
    if qrel_numeric != corpus_numeric:
        logger.error("❌ CRITICAL: ID format mismatch!")
        logger.error(f"   Qrel uses numeric: {qrel_numeric}")
        logger.error(f"   Corpus uses numeric: {corpus_numeric}")
        return False
    
    logger.info("="*60)
    logger.info("✅ All validation checks passed!")
    logger.info("="*60)
    
    # 生成诊断报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': {
            'query_qrel_overlap': len(common_queries) / len(query_ids_in_qrel) if query_ids_in_qrel else 0,
            'corpus_coverage': len(docs_in_corpus) / len(all_relevant_docs) if all_relevant_docs else 0,
            'query_mathml_coverage': queries_with_mathml / len(queries) if queries else 0,
            'total_queries': len(queries),
            'total_formulas': len(formulas),
            'total_relevant_pairs': sum(len(v) for v in relevance.values())
        },
        'id_samples': {
            'qrel_doc_ids': qrel_id_sample,
            'corpus_ids': corpus_id_sample
        }
    }
    
    report_file = data_dir / "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Validation report saved to {report_file}")
    
    return True

# ============================================================
# 🚀 核心3: 工作流编排器
# ============================================================
def run_complete_workflow():
    """
    执行完整的数据预处理工作流
    """
    logger.info("🚀 Starting complete MIR data preparation workflow...")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")
    
    # 定义工作流步骤
    steps = [
        WorkflowStep(
            name="Step 0: Corpus Preparation",
            script="scripts/prepare_final_arqmath.py",
            required_inputs=[
                "data/arqmath3/queries_arqmath3_task2.tsv",
                "data/arqmath3/qrel_task2_2022_official.tsv"
            ],
            outputs=[
                "data/processed/formulas.json",
                "data/processed/queries_full.json",
                "data/processed/relevance_labels.json"
            ]
        ),
        
        WorkflowStep(
            name="Step 1: Extract Query MathML from XML",
            script="scripts/extract_query_mathml_from_xml.py",
            required_inputs=[
                "data/arqmath3/Topics_Task2_2022_V0.1.xml",
                "data/processed/queries_full.json"
            ],
            outputs=[
                "data/processed/queries_full_with_mathml.json"
            ]
        ),
        
        WorkflowStep(
            name="Step 2: Supplement Missing MathML",
            script="scripts/fix_query_mathml_matching.py",
            required_inputs=[
                "data/processed/queries_full_with_mathml.json",
                "data/processed/formulas.json"
            ],
            outputs=[
                "data/processed/queries_final.json"
            ]
        )
    ]
    
    # 执行步骤
    results = []
    for step in steps:
        success = step.execute()
        results.append({
            'step': step.name,
            'success': success,
            'error': step.error_msg
        })
        
        if not success:
            logger.error(f"❌ Workflow stopped at: {step.name}")
            break
    
    # 如果所有步骤成功,执行验证
    all_success = all(r['success'] for r in results)
    
    if all_success:
        logger.info("🎉 All preprocessing steps completed!")
        logger.info("🔍 Running final validation...")
        
        validation_ok = validate_data_alignment()
        
        if validation_ok:
            logger.info("="*60)
            logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("   Data is ready for indexing and evaluation")
            logger.info("="*60)
            logger.info("📋 Next steps:")
            logger.info("   1. python scripts/build_index.py")
            logger.info("   2. python scripts/build_graph.py")
            logger.info("   3. python scripts/train_ranker.py")
            logger.info("   4. python scripts/run_eval.py")
            logger.info("="*60)
        else:
            logger.error("❌ Validation failed. Please review the errors above.")
    else:
        logger.error("❌ Workflow failed. Check logs above for details.")
    
    # 保存工作流报告
    workflow_report = {
        'timestamp': datetime.now().isoformat(),
        'steps': results,
        'overall_success': all_success
    }
    
    report_file = Path("data/processed/workflow_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(workflow_report, f, indent=2)
    
    logger.info(f"📄 Workflow report saved to {report_file}")
    
    return all_success

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    success = run_complete_workflow()
    sys.exit(0 if success else 1)