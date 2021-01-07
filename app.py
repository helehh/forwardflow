import streamlit as st
from forwardflow import ForwardFlow
import SessionState

from sklearn.decomposition import PCA
import pandas as pd
import altair as alt

PATH = "data/50k_skip_150_5_10_5.bin"

@st.cache
def load_embedding(allow_output_mutation=True):
    return ForwardFlow(PATH)

def main():
    session_state = SessionState.get(ff = None)

    st.title("Forward flow with word2vec embeddings")

    st.markdown("[Forward flow](http://www.forwardflow.org/) is a metric that measures semantic evolution of thoughts over time. It can be used to predict creativity.")
    
    st.markdown("In order to calculate forward flow of a thought stream the thoughts must be quantified somehow. In this simple demo we will be using [Estonian word2vec](https://datadoi.ee/handle/33/91) word embeddings to quantify thoughts (in contrast to the original paper where [LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis) was used).")

    st.subheader("Calculating forward flow")
    st.markdown("The formula itself is as follows:")

    st.latex(r'''
        (\sum_{i=2}^{n} \frac{\sum_{j=1}^{i-1} D_{i,j}}{i-1})/(n-1),
        ''')

    st.markdown("where n is the total number of thoughts within a stream and D_{i,j} is semantic\ndistance between two thoughts. Distance D as calculated as 1 - [cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity).")

    st.markdown("**Try it out yourself:**")

    if(session_state.ff == None):
        ff = load_embedding()
        session_state.ff = ff
    else:
        ff = session_state.ff

    words = st.text_input("Insert a list of Estonian words to calculate forward flow of the sequence. For example \"maja koer kask laps\".", value='', max_chars=150, key=None, type='default')
    wordlist = words.lower().split()

    can_ff = True
    if(len(wordlist) == 0):
        can_ff = False
    elif(len(wordlist) == 1):
        st.error("Insert at least 2 words")
        can_ff = False

    for word in wordlist:
        if(word not in ff.embedding):
            st.error("Word " + "'" + word + "'" + " not in vocabulary")
            can_ff = False

    if(can_ff):
        score = ff.score(wordlist)
        st.markdown("ForwardFlow: " + str(score))

    if(can_ff):
         #use PCA to project into 2d space and visualize
        st.markdown("**Visualization of word embeddings:**")

        pca = PCA(n_components=2)
        vectors = ff.embedding[wordlist]
        pca.fit(vectors)
        projected = pca.transform(vectors)

        source = pd.DataFrame({
        'PC1': projected[:, 0],
        'PC2': projected[:, 1],
        'label': wordlist})

        c = alt.Chart(source).mark_circle(color="firebrick", size=70
        ).encode(
            alt.X('PC1', scale=alt.Scale(domain=(-3, 3))),
            alt.Y('PC2', scale=alt.Scale(domain=(-3, 3))))

        text = c.mark_text(align='left', baseline='middle', dx=9
        ).encode(text='label')
        st.altair_chart(c+text)

if __name__ == '__main__':
	main()

