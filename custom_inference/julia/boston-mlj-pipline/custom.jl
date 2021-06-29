module Custom
using MLJ, DataFrames
using Base.Filesystem

export load_model, mypipe, score

DecisionTreeRegressor = @load DecisionTreeRegressor pkg="DecisionTree"
arb_imp = X -> coalesce.(X, -99999)
@pipeline FeatureSelector arb_imp ContinuousEncoder DecisionTreeRegressor name=mypipe

function load_model(code_dir)
    artifact_path = Filesystem.joinpath(code_dir, "tree_pipeline.jlso")
    model = machine(artifact_path)
    return model
end

function score(data, model; kwargs)
    predictions = predict(model, data)
    return DataFrame(Predictions = predictions)
end

end