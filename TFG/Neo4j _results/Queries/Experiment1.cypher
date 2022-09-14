//Train
MATCH (nodes:Individual) where nodes.Location = "SFLL" and nodes.Year in [1886, 1906, 1930, 1936]
MATCH (nodes)-[n:NAME]->(N:Name)
MATCH (nodes)-[sn:SURNAME]->(SN:Surname)
MATCH (nodes)-[ss:SECOND_SURNAME]->(SSN:Second_Surname)
MATCH (nodes)-[k:KIN]->(K:Kin)
MATCH (nodes)-[j:JOB]->(J:Job)
MATCH (nodes)-[sa:SAME_AS]->(SA:Individual) where (nodes.Year = 1886 and SA.Year = 1906 and SA.Location = "SFLL") OR (nodes.Year = 1930 and SA.Year = 1936 and SA.Location = "SFLL")

CALL gds.graph.create("Segon",
{
    nodes:{
        label:'Individual'
    },
    Value: {
        label: "Value",
        properties: ["n2v_embedding"]
    }
    //properties
    },
    
    {
    n:{
        type: "NAME",
        orientation: "UNDIRECTED"
    },
    sn:{
        type: "SURNAME",
        orientation: "UNDIRECTED"
    },
    ss:{
        type: "SECOND_SURNAME",
        orientation: "UNDIRECTED"
    },
    j:{
        type: "JOB",
        orientation: "UNDIRECTED"
    },
    k:{
        type: "KIN",
        orientation: "UNDIRECTED"
    },
    sa:{
        type: "SAME_AS",
        orientation: "UNDIRECTED"
    }
})

YIELD
  graphName, nodeProjection, nodeCount AS nod, relationshipCount AS rels
RETURN graphName, nod, rels


# Graph Sage Train:

CALL gds.beta.graphSage.train(
  $Name',
  {
    modelName: 'model',
    featureProperties: ['n2v_embedding'],
    projectedFeatureDimension: 64,
    activationFunction: 'sigmoid',
    randomSeed: 16,
    searchDepth : 3,
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
  
 -------------------------------------------------------------------------------------------------------------------------
 
 #### TEST 
 
 //Experiment2_test
MATCH (nodes:Individual) where nodes.Location = "SFLL" and nodes.Year in [1889, 1906,1930,1936]
MATCH (nodes)-[r]->(v:Value)
MATCH (nodes)-[sa:SAME_AS]->(SA:Individual) where (nodes.Year = 1889 and SA.Year = 1906 and SA.Location = "SFLL") 
OR ( nodes.Year = 1930 and SA.Year = 1936 and SA.Location = "SFLL")

CALL gds.graph.create($NAME,
{
    nodes:{
        label:'Individual'

    }, 
    Name: {
        label: "Name",
        properties: ["n2v_embedding"]
    },
    Surname: {
        label: "Surname",
        properties: ["n2v_embedding"]
    },    
    Second_Surname: {
        label: "Second_Surname",
        properties: ["n2v_embedding"]
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
    }
})
YIELD
  graphName, nodeProjection, nodeCount AS nod, relationshipCount AS rels
RETURN graphName, nod, rels




Call gds.beta.graphSage.write($NAME,{
    modelName: "whole",
    writeProperty: "$prop"
})





