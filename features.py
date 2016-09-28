from collections import Counter
from talking_tools import DirectTransformer, OneHotTransformer, ProjectionCountVectorizer


def device_model(x):
    # If device model is "pepe 4" will return "PEPE4"
    UNKWON = "UNKOWN"
    model = x["model"]
    if not model:
        return UNKWON
    model = model.split(" ")
    model = "".join(model)
    return model.upper()


def device_brand(x):
    UNKWON = "UNKOWN"
    brand = x["brand"]
    if not brand:
        return UNKWON
    return brand.upper()


class AmountOfApps(DirectTransformer):

    def transform_one(self, x):
        return len(x["apps_id"])


def _get_apps(x):
    return x["apps_id"]


class BagOfApps(OneHotTransformer):

    def __init__(self, func=_get_apps):
        self.f = func

    def fit(self, X, y=None):
        unseen = object()
        seen = set()
        for x in X:
            apps_id = self.f(x)
            for app_id in apps_id:
                seen.add(app_id)
        self.seen = list(sorted(seen)) + [unseen]
        return self

    def transform_one(self, x):
        result = [0] * len(self.seen)
        values = self.f(x)
        for value in values:
            if value in self.seen:
                result[self.seen.index(value)] = 1
            else:
                result[-1] = 1
        return result


class BagOfAppsVec(ProjectionCountVectorizer):
    UNKWON = "UNKOWN"

    def __init__(self):
        path = "apps_id"
        super().__init__(projection=path)

    def do_projection(self, doc):
        value = super().do_projection(doc)
        new = [x for x in value]
        result = " ".join(new)
        return result


class DeviceBrandFrecuency(DirectTransformer):

    def fit(self, X, y=None):
        counter = Counter([device_brand(x) for x in X])
        total = len(counter.values())
        self.seen = {brand: c / total for brand, c in counter.items()}
        return self

    def transform_one(self, x):
        brand = x["brand"].upper()
        result = self.seen.get(brand)
        return result


class DeviceModelFrecuency(DirectTransformer):

    def fit(self, X, y=None):
        counter = Counter([device_model(x) for x in X])
        total = len(counter.values())
        self.seen = {model: c / total for model, c in counter.items()}
        return self

    def transform_one(self, x):
        model = device_model(x)
        result = self.seen.get(model)
        return result


class DeviceBrand(ProjectionCountVectorizer):
    UNKWON = "UNKOWN"

    def __init__(self):
        path = "brand"
        super().__init__(projection=path)

    def do_projection(self, doc):
        value = super().do_projection(doc)
        if not value:
            return self.UNKWON
        return value.upper()


class DeviceModel(ProjectionCountVectorizer):
    UNKWON = "UNKOWN"

    def __init__(self):
        path = "model"
        super().__init__(projection=path)

    def do_projection(self, doc):
        model = super().do_projection(doc)
        if not model:
            return self.UNKWON
        model = model.split(" ")
        model = "".join(model)
        return model.upper()


class AppLabel(ProjectionCountVectorizer):
    UNKWON = "UNKOWN"

    def __init__(self):
        path = "labels"
        super().__init__(projection=path)

    def do_projection(self, doc):
        app_labes = super().do_projection(doc)
        if not app_labes:
            return self.UNKWON
        labels = " ".join(app_labes)
        return labels.upper()
