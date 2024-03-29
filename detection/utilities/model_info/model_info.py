import os
import jsonpickle
import datetime
import json


class ModelInfo:
    def __init__(self, author="Andrey Marakulin", compatible_det_info_json=None):
        """
        This class format collect full information about PyTorch for P10
        Recommended name for this class instance: model_info
        """
        super(ModelInfo, self).__init__()
        self.title = "Neural network model description format for P10"
        self.description = "This schema is computer-readable file for understand how to: prepare data," \
                           " load P10 elements classes and use threshold for PyTorch model"
        self.author = author
        self.model_classes = [21, 22, 23, 26, 24, 27, 29, 32, 31, 30, 17, 15, 16, 18, 19, 20, 8,
                              10, 13, 5, 9, 12, 11, 7, 14, 33, 34, 35, 36, 37, 38]

        self.schema = None
        self.compatible_det_info_json = compatible_det_info_json
        self.custom_name = None
        self.friendly_name = None
        self.train_params = None
        self.architecture = None

    def datetime_to_str(self, utc_shift_hours=3):
        delta = datetime.timedelta(hours=utc_shift_hours)  # MoscowUTC
        tzone = datetime.timezone(delta)
        return datetime.datetime.strftime(datetime.datetime.now(tzone), "%d-%m-%Y %H:%M")

    def datetime_to_datestr(self, utc_shift_hours=3):
        delta = datetime.timedelta(hours=utc_shift_hours)  # MoscowUTC
        tzone = datetime.timezone(delta)
        return datetime.datetime.strftime(datetime.datetime.now(tzone), "%d-%m-%Y")

    def datetime_to_modelname(self, utc_shift_hours=3):
        delta = datetime.timedelta(hours=utc_shift_hours)  # MoscowUTC
        tzone = datetime.timezone(delta)
        return datetime.datetime.strftime(datetime.datetime.now(tzone), "%Y-%m-%d")  # -%H-%M

    def create_compatible_det_info(self, det_dump_path="det_dump.json"):
        """
        How to create det_dump:
        det - is the Detector class in P10
        To dump it, you need write a few lines code below "det" in P10 and run:

        import jsonpickle
        dump = jsonpickle.encode(det)

        with open("det_dump.json", "w") as f:
            json.dump(data, f)
        """
        with open(det_dump_path) as det_dump:
            frozen = json.load(det_dump)
        det = jsonpickle.decode(frozen)

        with open("compatible_det_info.json", "w") as f:
            patterns_shapes = [pat.shape if pat is not None else None for pat in det.patterns]
            json.dump({"names": det.names, "parameters": det.parameters, "pat_orig": det.pat_orig,
                       "par_rotations": det.pat_rotations, "patterns_shapes": patterns_shapes}, f)

    def extended_classes(self, names, cl_id):
        same_classes = []
        name = str(names[cl_id]).replace("/0", "")
        for i in range(len(names)):
            if str(names[i]).replace("/0", "") == name:
                same_classes.append(i)
        same_classes = list(set(same_classes))
        return same_classes

    def generate_classes_groups(self, names, classes):
        nn_classes_ext = dict()
        for i, cl in enumerate(classes):
            nn_classes_ext[i + 1] = self.extended_classes(names, cl)
        return nn_classes_ext

    def set_author(self, author):
        self.author = author

    def set_classes(self, model_classes):
        self.model_classes = model_classes

    def set_custom_name(self, custom_name):
        self.custom_name = custom_name

    def set_friendly_name(self, friendly_name):
        self.friendly_name = friendly_name

    def set_architecture(self, architecture):
        if isinstance(architecture, str):
            self.architecture = architecture
        else:
            raise TypeError("PyTorch model json architecture must be string")

    def set_architecture_from_file(self, path):
        with open(path) as f:
            architecture = json.load(f)
        if isinstance(architecture, str):
            self.architecture = architecture
        else:
            raise TypeError("PyTorch model json architecture must be string")

    def compile_info(self):
        schema = dict()
        schema["title"] = self.title
        schema["author"] = self.author
        schema["description"] = self.description
        schema["createdate"] = {"datetime": self.datetime_to_str(), "date": self.datetime_to_datestr()}
        schema["modelname"] = self.custom_name if self.custom_name is not None \
            else "model_" + self.datetime_to_modelname() + '.pth'

        if self.compatible_det_info_json is None:
            try:
                with open("compatible_det_info.json") as f:
                    schema["compatible_det_info"] = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError("compatible_det_info not found. How to create"
                                        " it see in create_compatible_det_info function.")
        else:
            schema["compatible_det_info"] = self.compatible_det_info_json

        schema["classes"] = self.model_classes
        schema["classes_groups"] = self.generate_classes_groups(schema["compatible_det_info"]["names"],
                                                                schema["classes"])
        schema["classes_groups_list"] = list(set(schema["classes_groups"]))

        schema["architecture"] = self.architecture
        self.schema = schema

    def get_info(self):
        if self.schema is None:
            raise Exception("Model schema = None. It doesn't compiled or doesn't compiled correctly.")
        else:
            return self.schema

    def save_info(self, patch=".", custom_name=None):
        if self.schema is None:
            raise Exception("Model schema = None. It doesn't compiled or doesn't compiled correctly.")
        else:
            schema = self.get_info()
            name = custom_name if custom_name is not None else schema["modelname"]
            with open(os.path.join(patch, name + ".json"), "w") as f:
                json.dump(schema, f, indent=4)
                print('Info saved in: ', os.path.join(patch, name + ".json"))


if __name__ == "__main__":
    model_info = ModelInfo("Andrey Marakulin")
    model_info.compile_info()
    model_info.save_info()
    # print(json.dumps(model_info.get_info(), indent=4, sort_keys=True))
