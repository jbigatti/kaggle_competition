import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# From https://gist.github.com/jmansilla
class DirectTransformer:
    """Utility for building class-like features from a single-point function, but that may need
    some general configuration first (you usually override __init__ for that)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.transform_one(x) for x in X]).reshape((-1, 1))

    def transform_one(self, x):
        raise NotImplementedError


# From https://gist.github.com/jmansilla
class IndFunctionTransformer(DirectTransformer):
    """
    Utility for building transformers/features from a single-point function.
    """

    def __init__(self, func):
        self.f = func

    def transform_one(self, x):
        return self.f(x)


# From https://gist.github.com/jmansilla
class OneHotTransformer:
    def __init__(self, func):
        self.f = func

    def fit(self, X, y=None):
        unseen = object()
        seen = set()
        for x in X:
            seen.add(self.f(x))
        self.seen = list(sorted(seen)) + [unseen]
        return self

    def transform(self, X):
        return np.array([self.transform_one(x) for x in X])

    def transform_one(self, x):
        result = [0] * len(self.seen)
        value = self.f(x)
        if value in self.seen:
            result[self.seen.index(value)] = 1
        else:
            result[-1] = 1
        return result


# From https://gist.github.com/jmansilla
class ProjectionMixin:
    """Attributes projection base class"""
    def set_projection_path(self, path):
        self.projection_path = path.split('/')

    def do_projection(self, doc):
        for step in self.projection_path:
            if isinstance(doc, dict):
                doc = doc[step]
            elif isinstance(doc, (tuple, list)):
                if step.isdigit():
                    doc = doc[int(step)]
                else:   # only valid for namedtuples
                    doc = getattr(doc, step)
            else:
                raise ValueError('cant apply step %s' % step)
        return doc


# From https://gist.github.com/jmansilla
class ProjectionCountVectorizer(TfidfVectorizer, ProjectionMixin):
    """
    The usual CountVectorizer expects to work directly with strings.
    But what if you have dicts instead, and want to use some of its fields?
    You could do something like this:

        CountVectorizer(ngram_range=(1, 2), min_df=2,
                        preprocessor=lambda x: x["title"]),

    but that would have the following issues:
        - not pickleable, because of the lambda (fixable, but not very pretty)
        - by overriding "preprocessor", we are losing some of the CountVectorizer characteristics
          (like lowercasing or accents strip)

    So, instead, you can instanciate this count vectorizer like this:

        ProjectionCountVectorizer(ngram_range=(1, 2), min_df=2,
                                  projection="title")

    For several chained projections, use a slash for joining each path step, "like/this"
    Finally, for accessing lists or tuples by index, if any of the path steps is castable to int,
    will be tried like the number.
    """
    def __init__(self, projection, *args, **kwargs):
        self.set_projection_path(projection)
        super().__init__(*args, **kwargs)

    def build_preprocessor(self):
        built = super().build_preprocessor()

        def projection_and_preprocess(doc):
            return built(self.do_projection(doc))
        return projection_and_preprocess
