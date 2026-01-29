def ablate_pipeline(pipeline, flags: dict):
    """
    flags example:
    {
        "use_bert": True,
        "use_filter": True,
        "use_graph": False
    }
    """
    if not flags.get("use_graph", True):
        pipeline.graph_prior.pr = {}

    if not flags.get("use_filter", True):
        pipeline.filterer.apply = lambda q, c: c

    if not flags.get("use_bert", True):
        pipeline.coarse_ranker.feature_builder.build = (
            lambda qe, ce, c: [c.get("struct_score", 0.0)]
        )

    return pipeline


    消融实验控制器
