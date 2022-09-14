//Cosine and Euclidean Query
match (n:Individual)-[r:Candidate_Pairs]-(m:Individual) 
with n.$embedding$ as emb1, m.$embedding$ as emb2, r.class as gt
return n._id as src, m._id as trg, gds.alpha.similarity.cosine(emb1,emb2) as cos,
gds.alpha.similarity.euclidean(emb1,emb2) as eucl, gt



//Node_Sim Query

CALL gds.nodeSimilarity.stream($graph)
YIELD node1, node2, similarity
with gds.util.asNode(node1) as n1, gds.util.asNode(node2) as n2, similarity
where (n1.Year = 1910 and n2.Year = 1915) OR (n1.Year = 1924 and n2.Year = 1930)
match (n1)-[r:Candidate_Pairs]->(n2)
RETURN n1._id AS Person1, n2._id AS Person2, similarity, r.class
ORDER BY Person1


//Export CVS to make the sim calcualtion


