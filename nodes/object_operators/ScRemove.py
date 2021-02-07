import bpy

from bpy.props import PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode

class ScRemove(Node, ScObjectOperatorNode):
    bl_idname = "ScRemove"
    bl_label = "Remove"

    target_obj: PointerProperty(type=bpy.types.Object, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Target").init("target_obj", True)
    
    def error_condition(self):
        return(
            super().error_condition()
            or self.inputs["Target"].default_value == None
        )
    
    def functionality(self):
        bpy.data.objects.remove(self.inputs["Target"].default_value, do_unlink=True)
        #! SED: Better than selection semantics?
        # bpy.ops.object.delete()
