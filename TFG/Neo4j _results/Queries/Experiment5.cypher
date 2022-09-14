//Train
//whole_graph
MATCH (nodes:Individual) where nodes.Location = "SFLL" and nodes.Year in [1889, 1906,1930,1936]
MATCH (nodes)-[r]->(v:Value)
MATCH (nodes)-[sa:SAME_AS]->(SA:Individual) where (nodes.Year = 1889 and SA.Year = 1906 and SA.Location = "SFLL") 
OR ( nodes.Year = 1930 and SA.Year = 1936 and SA.Location = "SFLL")

CALL gds.graph.create("whole",
{
    nodes:{
        label:'Individual',
        properties: ["Cohort"]
    }, 
    v: {
        label: "Value",
        properties: ["n2v_embedding"]
    },
    Household: {
        label: "Household"
    },
    CensusHousehold:{
        label: "CensusHousehold",
        properties: ["Year"]
    }
    //properties
    },
    
    {
    NAME:{
        type: "NAME",
        orientation: "UNDIRECTED"
    },
    SURNAME:{
        type: "SURNAME",
        orientation: "UNDIRECTED"
    },
    SECOND_SURNAME:{
        type: "SECOND_SURNAME",
        orientation: "UNDIRECTED"
    },
    SAME_AS:{
        type: "SAME_AS",
        orientation: "UNDIRECTED"
    },
    KIN:{
        type: "KIN",
        orientation: "UNDIRECTED"
    },
    JOB:{
        type: "JOB",
        orientation: "UNDIRECTED"
    },
    IS_BUILDING:{
        type: "IS_BUILDING",
        properties: ["Year"]
    },
    VIU:{
        type: "VIU",
        orientation: "UNDIRECTED"
    },
    FAMILY:{
        type: "FAMILY",
        orientation: "UNDIRECTED"
    }
})
YIELD
  graphName, nodeProjection, nodeCount AS nod, relationshipCount AS rels
RETURN graphName, nod, rels


CALL gds.beta.graphSage.train(
  $Name',
  {
    modelName: 'model',
    featureProperties: ['n2v_embedding','Cohort', 'HISCLASS'],
    projectedFeatureDimension: 65,
    activationFunction: 'sigmoid',
    randomSeed: 16,
    searchDepth : 10,
    aggregator: "pool",
    sampleSizes: [25,10],
    epochs: 30 ,
    learningRate: 1e-4 
  }
)
YIELD modelInfo as info, trainMillis as ms
RETURN
  info.modelName as modelName,
  info.metrics.didConverge as didConverge,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses,
  ms
  
  
// TEST

//whole_graph_test
MATCH (nodes:Individual) where nodes.Location = "SFLL" and nodes.Year in [1910, 1915,1924,1930]
MATCH (nodes)-[r]->(v:Value)
MATCH (nodes)-[sa:SAME_AS]->(SA:Individual) where (nodes.Year = 1910 and SA.Year = 1915 and SA.Location = "SFLL") 
OR ( nodes.Year = 1924 and SA.Year = 1930 and SA.Location = "SFLL")

CALL gds.graph.create("whole_test",
{
    nodes:{
        label:'Individual',
        properties: ["Cohort"]
    }, 
    v: {
        label: "Value",
        properties: ["n2v_embedding"]
    },
    Household: {
        label: "Household"
    },
    CensusHousehold:{
        label: "CensusHousehold",
        properties: ["Year"]
    }
    //properties
    },
    
    {
    NAME:{
        type: "NAME",
        orientation: "UNDIRECTED"
    },
    SURNAME:{
        type: "SURNAME",
        orientation: "UNDIRECTED"
    },
    SECOND_SURNAME:{
        type: "SECOND_SURNAME",
        orientation: "UNDIRECTED"
    },
    Candidate_Pairs:{
        type: "Candidate_Pairs",
        orientation: "UNDIRECTED"
    },
    KIN:{
        type: "KIN",
        orientation: "UNDIRECTED"
    },
    JOB:{
        type: "JOB",
        orientation: "UNDIRECTED"
    },
    IS_BUILDING:{
        type: "IS_BUILDING",
        properties: ["Year"]
    },
    VIU:{
        type: "VIU",
        orientation: "UNDIRECTED"
    },
    FAMILY:{
        type: "FAMILY",
        orientation: "UNDIRECTED"
    }
})
YIELD
  graphName, nodeProjection, nodeCount AS nod, relationshipCount AS rels
RETURN graphName, nod, rels

Call gds.beta.graphSage.write($NAME,{
    modelName: "whole",
    writeProperty: "$prop"
})
  
