import sys
import bpy
import numpy
from numpy import array, uint32
import re

from mathutils import Vector, Matrix
from bpy.props import PointerProperty, StringProperty, EnumProperty, BoolProperty, IntProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import focus_on_object, remove_object, print_log

DEBUG_PRINT = False

PROP_WEIGHT_KEY = "weight"
PROP_LINK_KEY = "link"

# Debug util

def debug_log(title, str):
    if DEBUG_PRINT:
        print_log(title, str)

# DSL for writing grammars

class DslContext:
    def __init__(self, rand_state, property_prefix, start_link):
        # the only random selection state allowed
        self.rand_state = rand_state

        # any custom property will be assumed to have this prefix
        self.property_prefix = property_prefix

        # The following three values are mutually exclusive. We've either just created an object,
        # or created a virtual object from a definition with aliased links, or we've followed a link.

        # the current object in the production
        self.cur_obj = None
        # definitions must set aliases. if cur_obj_aliases != None, the cur_obj's children will
        # be ignored for following links, and we'll only use the aliases.
        self.cur_obj_aliases = None
        # the last link moved to
        self.cur_link = start_link

        # definitions at our disposal to search for
        self.definitions = {}
        # while defining, we can capture all exports for use in that definition during execution
        # that set of captures will be used for linking after the definition has executed
        self.export_captures = []

        # scopes let us do a production, and pop back to a current state
        self.scopes = []

        # list of all objects generated while running
        self.generated_objects = []

        # errors contain a list of errors found during the preparation or execution phases
        self.errors = []

    def print_state(self, title):
        if DEBUG_PRINT:
            debug_log("print_state", "%s: obj:%s obj_aliases:%s link:%s" % (title, repr(self.cur_obj), repr(self.cur_obj_aliases), repr(self.cur_link)))

    def add_generated_object(self, obj):
        self.generated_objects.append(obj)

    def set_current_object(self, obj):
        self.cur_obj = obj
        self.cur_obj_aliases = None
        self.cur_link = None
        self.print_state("set_current_object")

    def set_current_link(self, link):
        self.cur_link = link
        self.cur_obj = None
        self.cur_obj_aliases = None
        self.print_state("set_current_link")

    def get_current_link(self):
        return self.cur_link

    def get_custom_property_value(self, obj, partial_key):
        key = self.property_prefix + "_" + partial_key
        if obj != None and (key in obj.keys()):
            return obj[key]
        return None

    def get_matching_links(self, pattern):
        links = []
        if self.cur_obj_aliases:
            for alias in self.cur_obj_aliases.keys():
                if re.match(pattern, alias):
                    links.append(self.cur_obj_aliases[alias])
        elif self.cur_obj:
            for child in self.cur_obj.children:
                child_link_name = self.get_custom_property_value(child, PROP_LINK_KEY)
                if re.match(pattern, child_link_name):
                    links.append(child)
        return links

    def move_to_link(self, name):
        # if aliases defined, don't look at children
        if self.cur_obj_aliases:
            # if alias not found, do not default to children--it just wasn't defined.
            if name in self.cur_obj_aliases:
                self.set_current_link(self.cur_obj_aliases[name])
                self.print_state("move_to_link(alias)")
                return
        elif self.cur_obj:
            for child in self.cur_obj.children:
                child_link_name = self.get_custom_property_value(child, PROP_LINK_KEY)
                if child_link_name == name:
                    self.set_current_link(child)
                    self.print_state("move_to_link(child)")
                    return

        self.print_state("move_to_link(not found)")

    def set_definition(self, name, sequence):
        self.definitions[name] = sequence

    def get_definition(self, name):
        if name in self.definitions:
            return self.definitions[name]
        return None

    def push_scope(self):
        self.scopes.append({
            'cur_obj': self.cur_obj,
            'cur_obj_aliases': self.cur_obj_aliases,
            'cur_link': self.cur_link
        })
        self.print_state("push_scope")

    def get_scope(self):
        count = len(self.scopes)
        if count > 0:
            return self.scopes[count - 1]
        return None

    def pop_scope(self):
        top = self.scopes.pop()
        self.cur_obj = top['cur_obj']
        self.cur_obj_aliases = top['cur_obj_aliases']
        self.cur_link = top['cur_link']
        self.print_state("pop_scope")
        
    def push_define_export_capture(self):
        self.print_state("push_define_export_capture")
        self.export_captures.append({
            'aliases': {}
        })

    def _get_define_export_capture(self):
        count = len(self.export_captures)
        if count > 0:
            return self.export_captures[count - 1]
        return None

    def define_link_export(self, alias):
        cap = self._get_define_export_capture()
        if cap:
            cap['aliases'][alias] = self.cur_link

    def pop_define_export_capture(self):
        top = self.export_captures.pop()
        self.cur_obj = None
        self.cur_obj_aliases = top['aliases']
        self.cur_link = None
        self.print_state("pop_define_export_capture")

    def add_error(self, error):
        self.errors.append(error)

    def check_for_errors(self, title):
        for error in self.errors:
            print_log("Generate:", "%s: Error: %s" % (title, error,))

        return len(self.errors) > 0

    def clear_errors(self):
        self.errors = []

class DslValue:
    def __init__(self):
        pass

    def prepare(self, ctx):
        return True

    def execute(self):
        pass

class GnRange(DslValue):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def prepare(self, ctx):
        if not isinstance(self.min_value, int):
            ctx.add_error("GnRange: min_value not an integer")
            return False
        elif not isinstance(self.max_value, int):
            ctx.add_error("GnRange: max_value not an integer")
            return False
        elif self.min_value > self.max_value:
            ctx.add_error("GnRange: max_value is less than min_value")
            return False
        return True

    def evaluate(self, ctx):
        return ctx.rand_state.randint(self.min_value, self.max_value + 1)

# weights: a list of tuples where the first is the weight, the second is the value
# on evaluation, will use ctx random state to pick a value
class GnWeighted(DslValue):
    def __init__(self, weights):
        self.weights = weights

    def prepare(self, ctx):
        shape_error = False
        if not isinstance(self.weights, list) and not isinstance(self.weights, tuple):
            shape_error = True
        else:
            for weight in self.weights:
                if not isinstance(weight, list) and not isinstance(weight, tuple):
                    shape_error = True
                    break
        if shape_error:
            ctx.add_error("GnWeighted: weights must be tuple or list of tuple or list pairs")
            return False
        return True

    def evaluate(self, ctx):
        total = 0
        for w in self.weights:
            total += w[0]
        sum = 0
        thresh = total * ctx.rand_state.rand()
        value = None
        for w in self.weights:
            sum += w[0]
            if sum >= thresh:
                value = w[1]
                break
        
        if value == None and len(self.weights) > 0:
            value = self.weights[self.weights - 1]

        return value

# For a value to be evaluated as compilation (prepare) time
# Caches the result for use during the evaluate() call
class GnStatic(DslValue):
    def __init__(self, value):
        self.value = value

    def prepare(self, ctx):
        if isinstance(self.value, DslValue):
            self.value = self.value.evaluate(ctx)
            while isinstance(self.value, DslValue):
                self.value = self.value.evaluate(ctx)
        return True

    def evaluate(self, ctx):
        return self.value

# An operation in the DSL
# Contains helper methods
class DslOp:
    def __init__(self):
        pass

    def prepare(self, ctx):
        return True

    def prepare_value(self, ctx, value):
        if isinstance(value, DslValue):
            if not value.prepare(ctx):
                return False
        return True

    def evaluate_value(self, ctx, value):
        while isinstance(value, DslValue):
            value = value.evaluate(ctx)
        return value

    def move_hierarchy_collections(self, item, from_collection, to_collection):
        if from_collection != None:
            from_collection.objects.unlink(item)
        if to_collection != None:
            to_collection.objects.link(item)

        for child in item.children:
            self.move_hierarchy_collections(child, from_collection, to_collection)

    def move_hierarchy_to_link(self, ctx, obj, link=None):
        # move hierarchy to parent's collection
        # first remove the object's link to current collection
        obj_collection = None
        if len(obj.users_collection) > 0:
            obj_collection = obj.users_collection[0]        

        parent = ctx.get_current_link()
        if link != None:
            parent = link

        parent_collection = None
        if len(parent.users_collection) > 0:
            parent_collection = parent.users_collection[0]

        self.move_hierarchy_collections(obj, obj_collection, parent_collection)
        
        # final parenting within collection
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        parent.select_set(True)
        bpy.context.view_layer.objects.active = parent

        bpy.ops.object.parent_set(type="OBJECT", keep_transform=False)
        bpy.context.view_layer.objects.active = obj

# Define a named macro that can be executed as needed
# Definitions require explicit GnExportLink after the GnLink is made
class GnDefine(DslOp):
    def __init__(self, name, sequence):
        self.name = name
        self.sequence = sequence

    def prepare(self, ctx):
        if not isinstance(self.name, str):
            ctx.add_error("GnDefine: name of definition must be string literal")
            return False

        if not isinstance(self.sequence, list) and not isinstance(self.sequence, tuple):
            ctx.add_error("GnDefine: sequence must be list or tuple")
            return False

        for item in self.sequence:
            if not item.prepare(ctx):
                return False

        ctx.set_definition(self.name, self.sequence)
        return True

    def execute(self, ctx):
        pass

# A definition of the exported links in a command definition above
class GnExportLink(DslOp):
    def __init__(self, alias):
        self.alias = alias

    def prepare(self, ctx):
        return self.prepare_value(ctx, self.alias)

    def execute(self, ctx):
        alias = self.evaluate_value(ctx, self.alias)
        ctx.define_link_export(alias)

# A scope frame that contains a sequence. When finished, the context's object and link states return
# to where they were before the scope.
class GnScope(DslOp):
    def __init__(self, sequence):
        self.sequence = sequence

    def prepare(self, ctx):
        if not isinstance(self.sequence, list) and not isinstance(self.sequence, tuple):
            ctx.add_error("GnScope: scope sequence must be of type list or tuple")
            return False

        for item in self.sequence:
            if not item.prepare(ctx):
                return False

        return True

    def execute(self, ctx):
        ctx.push_scope()
        for item in self.sequence:
            item.execute(ctx)
        ctx.pop_scope()

# Repeat the contained sequence some number of times.
# Count can be any DslValue instance or integer constant
class GnRepeat(DslOp):
    def __init__(self, count, sequence):
        self.count = count
        self.sequence = sequence        

    def prepare(self, ctx):
        if not self.prepare_value(ctx, self.count):
            return False
        for item in self.sequence:
            if not item.prepare(ctx):
                return False
        return True

    def execute(self, ctx):
        count = self.evaluate_value(ctx, self.count)
        for _ in range(count):
            for item in self.sequence:
                item.execute(ctx)

# Create an instance of an object from a name, first looking in definitions, and then
# by looking for an identically named collection, and copying a weighted template object it
class GnInstance(DslOp):
    def __init__(self, name):
        self.name = name

    def prepare(self, ctx):
        return self.prepare_value(ctx, self.name)

    def execute(self, ctx):
        name = self.evaluate_value(ctx, self.name)
        sequence = ctx.get_definition(name)
        if sequence != None:
            ctx.push_define_export_capture()
            for item in sequence:
                item.execute(ctx)
            ctx.pop_define_export_capture()
        elif name in bpy.data.collections:
            selected_obj = None
            src_collection = bpy.data.collections[name]
            src_objects = []
            if src_collection != None:                            
                for ch in src_collection.objects:
                    # check for top level
                    if ch.parent == None:
                        src_objects.append(ch)
            weight_total = 0
            weight_pairs = []
            if len(src_objects) > 0:                
                for candidate in src_objects:
                    weight = ctx.get_custom_property_value(candidate, PROP_WEIGHT_KEY)
                    if type(weight) not in [int, float]:
                        weight = 1.0
                    weight_total += weight
                    weight_pairs.append((candidate, weight))
                rand01 = ctx.rand_state.rand()
                weight_thresh = rand01 * weight_total
                weight_sum = 0

                for pair in weight_pairs:
                    weight_sum += pair[1]
                    if weight_sum > weight_thresh:
                        selected_obj = pair[0]
                        break
                
                if selected_obj == None:                
                    selected_obj = src_objects[len(src_objects) - 1]

            if selected_obj != None:
                # hierarchically duplicate object
                bpy.ops.object.select_all(action="DESELECT")
                selected_obj.select_set(True)
                bpy.context.view_layer.objects.active = selected_obj
                bpy.ops.object.select_grouped(extend=True, type='CHILDREN_RECURSIVE')
                bpy.ops.object.duplicate()

                copy = bpy.context.view_layer.objects.active
                if copy != None:
                    copy.matrix_world = ctx.get_current_link().matrix_world.copy()

                    # move to parent's collection
                    self.move_hierarchy_to_link(ctx, copy)

                    ctx.set_current_object(copy)
                    ctx.add_generated_object(copy)

class GnCopyParentLinkChildren(DslOp):
    def __init__(self, link, mirror=None):
        self.link = link
        self.mirror = mirror

    def prepare(self, ctx):
        return self.prepare_value(ctx, self.link)

    def execute(self, ctx):
        src_link_name = self.evaluate_value(ctx, self.link)
        dest_link = ctx.get_current_link()
        if dest_link != None:
            parent = dest_link.parent
            if parent != None:
                for child in parent.children:
                    child_link_name = ctx.get_custom_property_value(child, PROP_LINK_KEY)
                    if child_link_name == src_link_name:
                        for gch in child.children:
                            # hierarchically duplicate all children
                            bpy.ops.object.select_all(action="DESELECT")
                            gch.select_set(True)
                            bpy.context.view_layer.objects.active = gch
                            bpy.ops.object.select_grouped(extend=True, type='CHILDREN_RECURSIVE')
                            bpy.ops.object.duplicate()

                            gch_copy = bpy.context.view_layer.objects.active
                            if gch_copy != None:
                                gch_copy.matrix_world = dest_link.matrix_world.copy()

                                # move to parent's collection
                                self.move_hierarchy_to_link(ctx, gch_copy, dest_link)

                                mirror_tuple = None
                                scale_tuple = None
                                if self.mirror == 'x':
                                    mirror_tuple = (True, False, False)
                                    scale_tuple = (1, 0, 0)
                                elif self.mirror == 'y':
                                    mirror_tuple = (False, True, False)
                                    scale_tuple = (0, 1, 0)
                                elif self.mirror == 'z':
                                    mirror_tuple = (False, False, True)
                                    scale_tuple = (0, 0, 1)

                                #if mirror_tuple != None:
                                if scale_tuple != None:
                                    bpy.ops.object.select_all(action="DESELECT")
                                    gch_copy.select_set(True)
                                    bpy.context.view_layer.objects.active = gch_copy

                                    mat_mirror = Matrix.Scale(-1, 4, scale_tuple)
                                    gch_copy.matrix_local = mat_mirror @ gch_copy.matrix_local
                                   
                                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                                    bpy.ops.object.editmode_toggle()
                                    bpy.ops.mesh.select_all(action='SELECT')
                                    bpy.ops.mesh.flip_normals()
                                    bpy.ops.object.mode_set()

                                ctx.add_generated_object(gch_copy)

# In the state machine, move from the last made object to a named link
# If the last named object was a dictionary defined macro, it will search the current exports.
# otherwise, it will look for children with a <prefix>_link custom property where the value 
# matches the link name.
class GnLink(DslOp):
    def __init__(self, name):
        self.name = name

    def prepare(self, ctx):
        return self.prepare_value(ctx, self.name)

    def execute(self, ctx):
        name = self.evaluate_value(ctx, self.name)
        ctx.move_to_link(name)

# In the state machine, move from the last made object to a named link
# If the last named object was a dictionary defined macro, it will search the current exports.
# otherwise, it will look for children with a <prefix>_link custom property where the value 
# matches the link name.
class GnEachLink(DslOp):
    def __init__(self, pattern, sequence):
        self.pattern = pattern
        self.sequence = sequence

    def prepare(self, ctx):
        if not isinstance(self.pattern, str):
            ctx.add_error("GnEachLink: search pattern must be string literal")
            return False
        if not isinstance(self.sequence, list) and not isinstance(self.sequence, tuple):
            ctx.add_error("GnEachLink: scope sequence must be of type list or tuple")
            return False
        for item in self.sequence:
            if not item.prepare(ctx):
                return False
        return True

    def execute(self, ctx):                
        links = ctx.get_matching_links(self.pattern)
        for link in links:
            ctx.push_scope()
            ctx.set_current_link(link)
            for item in self.sequence:
                item.execute(ctx)
            ctx.pop_scope()

# Generator node

class ScGenerate(Node, ScObjectOperatorNode):
    bl_idname = "ScGenerate"
    bl_label = "Generate"

    prop_random_state: StringProperty()
    in_file: StringProperty(subtype='FILE_PATH', update=ScNode.update_value)
    in_prefix: StringProperty(default="gen", update=ScNode.update_value)
    in_seed: IntProperty(default=0, min=0, update=ScNode.update_value)
    in_max_depth: IntProperty(default=10, min=0, update=ScNode.update_value)
    in_clear_children: BoolProperty(default=True, update=ScNode.update_value)
    prop_obj_array: StringProperty(default="[]")

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketString", "File").init("in_file", True)
        self.inputs.new("ScNodeSocketString", "Prefix").init("in_prefix", False)
        self.inputs.new("ScNodeSocketNumber", "Seed").init("in_seed", True)
        self.inputs.new("ScNodeSocketNumber", "Max Depth").init("in_max_depth", True)
        self.inputs.new("ScNodeSocketBool", "Clear Children").init("in_clear_children", False)
        self.outputs.new("ScNodeSocketArray", "Generated Objects")
        
    def error_condition(self):
        return (
            super().error_condition()
            or self.inputs["File"].default_value == ""
        )
    
    def pre_execute(self):
        super().pre_execute()
        self.prop_obj_array = "[]"

    def functionality(self):
        pass
        # 
    
    def post_execute(self):
        seed = abs(self.inputs["Seed"].default_value)

        rs = numpy.random.RandomState(int(seed))
        if (not self.first_time):
            debug_log("post_execute:", "RS:" + str(self.prop_random_state))
            rs.set_state(eval(self.prop_random_state))

        generated_objects = []

        file_text = ''
        try:
            file_path = self.inputs["File"].default_value
            file_path = bpy.path.abspath(self.inputs["File"].default_value)
            fd = open(file_path, "r")
            file_text = fd.read()
            fd.close()
        except:
            ex = sys.exc_info()[0]
            print_log("ScGenerate", "There was an error loading the file: %s" % (ex,))

        debug_log("ScGenerate", "Read %d characters from file %s" % (len(file_text), file_path))

        if isinstance(file_text, str) and file_text != "":
            start_obj = self.inputs["Object"].default_value
            if start_obj != None:
                # clear start object's children
                if self.inputs["Clear Children"].default_value:
                    for child in start_obj.children:
                        remove_object(child, hierarchy=True)

                # root of rule recursion
                dsl_ctx = DslContext(rs, self.inputs["Prefix"].default_value, start_obj)

                op_list = eval(file_text)
                for op in op_list:
                    op.prepare(dsl_ctx)

                    if dsl_ctx.check_for_errors("Preparation"):
                        break

                    op.execute(dsl_ctx)
                    if dsl_ctx.check_for_errors("Execution"):
                        break

                generated_objects = dsl_ctx.generated_objects

        self.prop_obj_array = repr(generated_objects)
        self.prop_random_state = repr(rs.get_state())

        out = super().post_execute()
        out["Generated Objects"] = self.prop_obj_array
        return out
    
    def free(self):
        for object in self.prop_obj_array[1:-1].split(', '):
            try:
                obj = eval(object)
            except:
                print_log(self.id_data.name, self.name, "free", "Invalid object: " + object)
                continue
            self.id_data.unregister_object(obj)