# Some Notes

In order to get comfy to accept two types of inputs here's an example used by KJ nodes:

```Python
class ImageConcanate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # image1 drives the output type; image2 can independently be IMAGE or MASK and gets
        # converted to image1's type inside concatenate().
        type_template = io.MatchType.Template("image_or_mask", allowed_types=[io.Image, io.Mask])
        return io.Schema(
            node_id="ImageConcanate",
            category="KJNodes/image",
            description=(
                "Concatenates image2 to image1 in the specified direction.\n"
                "Both inputs accept IMAGE or MASK; the output type follows image1.\n"
                "If image2 is a different type than image1 it's converted (RGB mean for image→mask,\n"
                "channel-replicate for mask→image).\n"
                "When match_image_size is False and dimensions don't match along the shared axis,\n"
                "the smaller image is centered and zero-padded instead of erroring."
            ),
            inputs=[
                io.MatchType.Input("image1", template=type_template),
                io.MultiType.Input("image2", types=[io.Image, io.Mask]),
                io.Combo.Input("direction", options=['right', 'down', 'left', 'up'], default='right'),
                io.Boolean.Input("match_image_size", default=True),
            ],
            outputs=[
                io.MatchType.Output(template=type_template, display_name="output"),
            ],
        )
    ...
```