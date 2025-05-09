import textwrap

EXAMPLE_CODE = textwrap.dedent(
    """\
    from manim import *
    from manim4ai import LayoutManager  # Use your actual module name here
    import numpy as np

    def transformation_circle(m):
        return m.move_to(LEFT * 3).rotate(PI)

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
        "animation_function": transformation_circle,
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
    - Do not modify the `construct` method; it must remain empty.
    - Always use the `place` method of `LayoutManager` to animate objects.
    - `animation_function` defined with `def` or `lambda` must accept a single `Mobject` as its input.
    - `animation_function` can also be a `manim.Animation` object like `FadeOut`.
    - Pass `animation_function` as a parameter without calling it.
    - Do not define custom functions or use variables outside those shown in the example.
    - Animations should be visual, comprehensive, information dense and intuitive.
    - No object should collide with others unless they share a `collision_group`.
      For example, if you are simulating objects on a NumberPlane, the NumberPlane 
      should have the same `collision_group` as the other two objects etc.
    - All objects must remain fully visible within the screen bounds.
    - Use up to 3 objects in a per animation order.
    - Avoid redundant imports or statements.
    - Aim for longer videos.
    - Objects with different `animation_order` can't be on the same screen.
    - Objects should be added only through the `place` method of the `LayoutManager`.
"""
)

LAYOUT_DOC = textwrap.dedent(
    """\
    Here are the methods of the layout manager class you can use:
    
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

    return textwrap.dedent(
        f"""\
        The code you generated contains errors.

        Possible issues include:
        - Syntax errors (e.g., incorrect method calls or object definitions)
        - Objects moving outside the visible scene area
        - Objects unintentionally colliding or overlapping
        - Violating the animation or layout constraints

        You can resolve these issues by checking the syntatic mistakes,
        changing the sizes of the objects, changing the paths slightly to avoid conflicts.

        Please regenerate the code, fixing the specific problems below:

        {issues.strip()}

        Return only the corrected Python code.
        Do not include comments, explanations, or any other text.
    """
    )
