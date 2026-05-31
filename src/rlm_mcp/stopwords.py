"""
Stopwords PT+EN para tokenização BM25 do RLM MCP Server.

Termos de altíssima frequência e baixo poder discriminativo. Removidos na
tokenização do índice invertido e da query para que o BM25 não gaste IDF/peso
em palavras funcionais. Expande o `_QUERY_STOPWORDS` mínimo do dampening
(indexer.py) — aqui a lista é maior porque o BM25 indexa TODO o vocabulário do
corpus, não só os termos da query.

Nota de domínio (corpus biomédico/espiritual do autor): mantemos FORA da lista
termos curtos que são conteúdo no domínio (ex: "rim", "osso", "dor", "ser",
"luz", "deus"). Stopword é função gramatical, não palavra curta.
"""

# Português (BR) — artigos, preposições, conjunções, pronomes, auxiliares,
# advérbios de alta frequência. NFKD/acento-folding é aplicado na tokenização,
# então registramos as formas SEM acento (o tokenizer compara já dobrado).
_STOPWORDS_PT = {
    # artigos
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    # preposições + contrações
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "ao", "aos", "a", "as", "num", "numa", "nuns", "numas",
    "pelo", "pela", "pelos", "pelas", "por", "para", "pra", "pro",
    "com", "sem", "sob", "sobre", "ate", "apos", "ante", "entre", "contra",
    "desde", "perante", "tras", "dele", "dela", "deles", "delas",
    "deste", "desta", "destes", "destas", "desse", "dessa", "desses", "dessas",
    "daquele", "daquela", "disto", "disso", "daquilo", "nesse", "nessa",
    "neste", "nesta", "nele", "nela", "neles", "nelas", "naquele", "naquela",
    # conjunções
    "e", "ou", "mas", "porem", "contudo", "todavia", "entretanto",
    "que", "porque", "pois", "se", "como", "quando", "enquanto", "embora",
    "logo", "portanto", "assim", "tambem", "nem", "ja", "ainda",
    # pronomes
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas", "voce", "voces",
    "me", "te", "lhe", "lhes", "se", "nos", "vos", "mim", "ti", "si",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas", "nosso", "nossa", "nossos", "nossas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isto", "isso", "aquilo",
    "qual", "quais", "quem", "cujo", "cuja", "cujos", "cujas",
    "algum", "alguma", "alguns", "algumas", "nenhum", "nenhuma",
    "todo", "toda", "todos", "todas", "outro", "outra", "outros", "outras",
    "mesmo", "mesma", "mesmos", "mesmas", "tal", "tais",
    # verbos auxiliares / cópula de alta frequência (formas comuns)
    "ser", "sou", "es", "somos", "sao", "era", "eram", "foi", "foram",
    "sera", "serao", "seja", "sejam", "sendo", "sido",
    "estar", "estou", "esta", "estamos", "estao", "estava", "estavam",
    "esteve", "estiveram", "estará", "estara", "esteja",
    "ter", "tenho", "tem", "temos", "tinha", "tinham", "teve", "tiveram",
    "havia", "haviam", "houve", "ha", "hao",
    "ir", "vai", "vao", "foi", "foram", "vou", "vamos",
    "poder", "pode", "podem", "podia", "podiam", "pode",
    "fazer", "faz", "fazem", "fez", "fazia",
    # advérbios / quantificadores funcionais
    "nao", "sim", "muito", "muita", "muitos", "muitas", "mais", "menos",
    "bem", "mal", "so", "apenas", "tao", "quanto", "quao", "onde", "aonde",
    "aqui", "ali", "la", "ca", "agora", "entao", "depois", "antes", "sempre",
    "nunca", "talvez", "cada", "qualquer", "quaisquer",
}

# English — articles, prepositions, conjunctions, pronouns, auxiliaries, common
# adverbs. Corpus do autor mistura PT/EN (ReCODE/Bredesen em inglês).
_STOPWORDS_EN = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "such",
    "some", "any", "no", "every", "each", "all", "both", "few", "many", "much",
    "other", "another", "same",
    # prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "onto", "upon", "within", "without",
    "per", "via",
    # conjunctions
    "and", "or", "but", "nor", "so", "yet", "because", "as", "until",
    "while", "if", "though", "although", "unless", "whereas",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "themselves", "who", "whom", "whose", "which",
    "what", "whatever", "whoever",
    # auxiliaries / copula
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "ought",
    # high-frequency adverbs / fillers
    "not", "only", "very", "too", "also", "just", "even", "more", "most",
    "less", "least", "here", "there", "where", "when", "why", "how",
    "now", "than", "ever", "never", "always", "often", "still", "yes",
}

# União exposta. Set único minimiza lookups na tokenização (hot path).
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_PT | _STOPWORDS_EN)
