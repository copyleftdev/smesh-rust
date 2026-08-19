//! Deterministic PDF rendering for fixtures and the Meridian corpus.
//!
//! lopdf with rayon off and no Info dictionary (no timestamps) renders
//! byte-identical output for identical input — which keeps `DocId`s stable
//! across runs, the property the whole content-addressed pipeline leans on.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Document, Object, Stream};

/// Render one page per entry; each entry is the page's lines. Text must be
/// plain ASCII prose (Helvetica, standard encoding) — all Meridian needs.
pub fn render_pdf(pages: &[Vec<String>]) -> Vec<u8> {
    let mut doc = Document::with_version("1.4");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });

    let mut kids: Vec<Object> = Vec::new();
    for lines in pages {
        let mut operations = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("TL", vec![16.into()]),
            Operation::new("Td", vec![72.into(), 720.into()]),
        ];
        for line in lines {
            operations.push(Operation::new(
                "Tj",
                vec![Object::string_literal(line.as_str())],
            ));
            operations.push(Operation::new("T*", vec![]));
        }
        operations.push(Operation::new("ET", vec![]));

        let content = Content { operations };
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            content.encode().expect("static content encodes"),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        kids.push(page_id.into());
    }

    let count = i64::try_from(kids.len()).expect("page count fits i64");
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).expect("in-memory save cannot fail");
    bytes
}
