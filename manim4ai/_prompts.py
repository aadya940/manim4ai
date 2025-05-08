import textwrap

EXAMPLE_CODE = textwrap.dedent(
    """\
    from manim import *
    from manim4ai import LayoutManager  # Use your actual module name here
    import numpy as np

    class DummyScene(Scene):
        def construct(self): 
            pass

    scene = DummyScene()
    layout = LayoutManager(max_objects=3, samples=20)

    square = Square().shift(LEFT * 3)
    circle = Circle().shift(RIGHT * 3)
    triangle = Triangle().shift(UP * 2)

    layout.place(scene, square, {
        "animation_order": 0,
        "animation_function": lambda m: m.shift(RIGHT * 6),
        "id": "square",
        "collision_group": "g1",
        "duration": 2
    })

    layout.place(scene, circle, {
        "animation_order": 0,
        "animation_function": lambda m: m.move_to(LEFT * 3).rotate(PI),
        "id": "circle",
        "collision_group": "g2",
        "duration": 2
    })

    layout.place(scene, triangle, {
        "animation_order": 0,
        "animation_function": lambda m: m.scale(1.5).rotate(PI/2).move_to(DOWN * 2),
        "id": "triangle",
        "collision_group": "g3",
        "duration": 2
    })
"""
)

LIBRARIES = "- manim\n- manim4ai\n- numpy"

GUIDELINES = textwrap.dedent(
    """\
    Guidelines:
        - The animations should be intuitive and easy to understand.
        - It should be no longer than 2 minutes.
        - Make sure the objects are correctly placed and animated in the animation.
        - Only return the python code, nothing else strictly.
"""
)

LAYOUT_DOC = textwrap.dedent(
    """\
    Here are some important functions of the layout manager class:
    
    - LayoutManager.place:
        Place an object in the scene with its animation configuration.
        
        Args:
            session (Scene): The Manim scene
            object (Mobject): The object to animate
            animation_config (dict): Configuration including:
                - animation_order: Order in which animations should play
                - animation_function: Function that transforms the object
                - collision_group (optional): Group ID for collision exemption
                - id (optional): Unique identifier for the object
                - duration (optional): Animation duration in seconds
    
    - LayoutManager.static_check:
        Performs static checks on all placed objects and returns a string report.
    
    - LayoutManager.__init__:
        Initialize a new LayoutManager.
        Args:
            max_objects (int): Maximum number of objects in one animation order
            samples (int): Trajectory approximation accuracy
"""
)


def system_prompt(topic: str) -> str:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Expected a non-empty string as topic.")

    return textwrap.dedent(
        f"""\
        You are a programmer and you have to generate an animation script on the following topic:

        {topic}

        You are allowed to use the following libraries in the animation code:
        {LIBRARIES}

        Here is an example on how to implement an animation:

        {EXAMPLE_CODE}

        {LAYOUT_DOC}

        {GUIDELINES}
    """
    )


def build_prompt(issues: str) -> str:
    if not isinstance(issues, str):
        raise ValueError("Issues must be a string.")
    return f"Please regenerate the code.\n Only return the code.\n Your generated code has the following issues:\n\n{issues.strip()}"
