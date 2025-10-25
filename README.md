## *Job Recommendation System*

Recommendation systems are AI-based tools that analyze user behavior, preferences, and past interactions to suggest relevant items like products, movies, or songs. They are essential for personalizing user experiences, increasing engagement, and boosting sales. Commonly used in platforms like Netflix (movie suggestions), Amazon (product recommendations), and Spotify (music playlists), they help users discover content tailored to their interests efficiently.

In this project I have made a Job Recommendation System, high level workflow is like a user uploades its resume and it will be recommended with top 10 jobs.

## *Methodology*
I went through two approaches for this problem statement
1) Two Tower Model:- This is quite good and scalable architecture for recommendation, but this didn't provide good results in our case because of the following reasons:
    * I did not have user-job interaction data
    * Applying clustering resulted in a huge overlap between different job domains
    * It was difficult to figure out "Hard negatives" in the dataset
    * As a result even though the model was trained it resulted in overfitting
    So due to these reasons I dropped this approach and came up with the 2nd approach

2) Hybrid search with domain re-ranking: This is quite a simple pipline, in crucs first user uploads his/her resume then the content is extracted and converted into a embedding at first hybrid search is performed then jobs are re-ranked with help of a domain classfication model. So how this solved the previous problems:
    * Firstly no need of "hard negatives" here
    * Hybrid search is focused on both keyword and vector matching which focuses on most relavant jobs on top of that this is fast so inference time will also be less
    * Domain re-ranker is used to show the most relevant jobs first, this model is trained on around 2 lakh data samples where contribution of data from all domains is equal i.e no imbalance class.
    * During re-ranking diversity of jobs is also assured 

